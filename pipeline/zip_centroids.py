"""Build data/zip_centroids.csv from the US Census Gazetteer ZCTA file.

Why: the Census file is public domain and gives each ZIP area's center point,
which is all we need for "trials within N miles". Run once:
    python -m pipeline.zip_centroids
"""
from __future__ import annotations

import csv
import io
import zipfile

import httpx

from backend import config


def convert(zip_bytes: bytes) -> list[tuple[str, float, float]]:
    """Parse the zipped tab-separated Gazetteer into (zip, lat, lon) rows."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        text = zf.read(name).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = [h.strip() for h in next(reader)]
    i_zip, i_lat, i_lon = header.index("GEOID"), header.index("INTPTLAT"), header.index("INTPTLONG")
    rows = []
    for r in reader:
        if len(r) <= max(i_zip, i_lat, i_lon):
            continue
        rows.append((r[i_zip].strip().zfill(5), float(r[i_lat]), float(r[i_lon])))
    return rows


def main() -> None:
    resp = httpx.get(config.ZIP_GAZETTEER_URL, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    rows = convert(resp.content)
    config.ZIP_CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ZIP_CENTROIDS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zip", "lat", "lon"])
        w.writerows(rows)
    print(f"wrote {len(rows)} ZIPs to {config.ZIP_CENTROIDS_PATH}")


if __name__ == "__main__":
    main()
