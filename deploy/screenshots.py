"""Headless-Chrome click-through that saves screenshots to docs/screenshots/ (stand-in for a human test).

Needs: Chrome installed, `websockets` (comes with uvicorn[standard]), the app on :8000 and the mock server on :8765
(run_api.ps1 and run_frontend.ps1). Usage: python deploy/screenshots.py
"""
import sys, asyncio, base64, json, subprocess, tempfile, time, urllib.request
from pathlib import Path
import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SIZES = {"desktop": (1280, 900), "phone": (375, 812)}
LIVE, MOCK = "http://127.0.0.1:8000/", "http://127.0.0.1:8765/"


class Page:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        await self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.i:
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def size(self, w, h):
        await self.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=w < 500)

    async def goto(self, url, wait=1.5):
        await self.call("Page.navigate", url="about:blank")  # hash-only navigation would keep the old DOM state
        await asyncio.sleep(0.3)
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def shot(self, name, full=True):
        # Resize to content height so full-page captures include everything.
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=full)
        (OUT / name).write_bytes(base64.b64decode(r["data"]))

    async def pdf(self, name):
        r = await self.call("Page.printToPDF", printBackground=True)
        (OUT / name).write_bytes(base64.b64decode(r["data"]))


async def scenario(p: Page, base: str, tag: str, mock: bool):
    for dev, (w, h) in SIZES.items():
        await p.size(w, h)
        q = "?mock=1" if mock else ""
        await p.goto(base + q + "#"); await p.shot(f"{tag}-landing-{dev}.png", full=False)
        await p.js("document.getElementById('q').value='breast cancer';document.getElementById('search-form').requestSubmit()")
        await asyncio.sleep(2.5); await p.shot(f"{tag}-results-{dev}.png")
        await p.goto(base + q + "#"); await p.js("document.getElementById('q').value='pizza recipes';document.getElementById('search-form').requestSubmit()")
        await asyncio.sleep(2.5); await p.shot(f"{tag}-offtopic-{dev}.png")
        await p.goto(base + q + "#"); await p.js("document.getElementById('q').value='breast cancer';document.getElementById('search-form').requestSubmit()")
        await asyncio.sleep(2.5)
        await p.js("document.querySelector('#results .card .btn.primary').click()")
        await asyncio.sleep(3); await p.shot(f"{tag}-summary-{dev}.png")
        if dev == "desktop":
            await p.pdf(f"{tag}-summary-print.pdf")


async def about_scenario(p: Page):
    """Mock-mode screenshots of the 'About you' form, match badges and the new study panel sections."""
    for dev, (w, h) in SIZES.items():
        await p.size(w, h)
        await p.goto(MOCK + "?mock=1#")
        await p.js("document.getElementById('about-you').open=true;document.getElementById('age').value='45';"
                   "document.getElementById('sex').value='female';"
                   "document.getElementById('conditions').value='HER2 positive, pregnant, had chemotherapy'")
        await p.shot(f"about-form-{dev}.png", full=False)
        await p.js("document.getElementById('q').value='breast cancer';document.getElementById('search-form').requestSubmit()")
        await asyncio.sleep(2.5); await p.shot(f"about-results-{dev}.png")
        await p.js("document.querySelector('#results .card .btn.primary').click()")
        await asyncio.sleep(3); await p.shot(f"about-summary-{dev}.png")
        print(dev, "overflow:", await p.js("document.documentElement.scrollWidth > document.documentElement.clientWidth"))


async def toggle_scenario(p: Page):
    """Mock-mode screenshots of the criteria panel in both wording states (original default, then simplified)."""
    for dev, (w, h) in SIZES.items():
        await p.size(w, h)
        await p.goto(MOCK + "?mock=1#")
        await p.js("document.getElementById('about-you').open=true;document.getElementById('age').value='45';"
                   "document.getElementById('sex').value='female';"
                   "document.getElementById('conditions').value='HER2 positive, pregnant, had chemotherapy'")
        await p.js("document.getElementById('q').value='breast cancer';document.getElementById('search-form').requestSubmit()")
        await asyncio.sleep(2.5)
        await p.js("document.querySelector('#results .card .btn.primary').click()")
        await asyncio.sleep(3)
        await p.js("document.querySelector('.btn.toggle').scrollIntoView()")
        await p.shot(f"criteria-original-{dev}.png", full=False)
        await p.js("document.querySelector('.btn.toggle').click()")
        await asyncio.sleep(0.5); await p.shot(f"criteria-simplified-{dev}.png", full=False)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={prof}",
                             "--no-first-run", "--disable-gpu", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                break
            except Exception:
                time.sleep(0.5)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=64_000_000) as ws:
            p = Page(ws)
            await p.call("Page.enable"); await p.call("Runtime.enable")
            if "--toggle" in sys.argv:
                await toggle_scenario(p)
                return
            if "--about" in sys.argv:
                await about_scenario(p)
                return
            await scenario(p, MOCK, "mock", True)
            await scenario(p, LIVE, "live", False)
            # live 404 path: a real-looking NCT with no stored summary
            for dev, (w, h) in SIZES.items():
                await p.size(w, h)
                await p.goto(LIVE)
                await p.js("openSummary('NCT00000000', document.body)")
                await asyncio.sleep(2.5); await p.shot(f"live-summary-missing-{dev}.png")
            # overflow + focus checks
            await p.size(375, 812); await p.goto(LIVE)
            print("phone horizontal overflow:", await p.js("document.documentElement.scrollWidth > document.documentElement.clientWidth"))
    finally:
        proc.terminate()


asyncio.run(main())
