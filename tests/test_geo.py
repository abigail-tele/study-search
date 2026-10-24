import pytest

from backend import geo

NYC, LA = (40.7128, -74.0060), (34.0522, -118.2437)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("zip,lat,lon\n10001,40.7506,-73.9972\n90001,33.9731,-118.2479\n02134,42.3573,-71.1338\n")
    geo.load_centroids.cache_clear()
    return p


def test_haversine_nyc_la():
    assert geo.haversine_miles(NYC, LA) == pytest.approx(2445, rel=0.01)


def test_haversine_zero():
    assert geo.haversine_miles(NYC, NYC) == 0


def test_lookup(csv_path):
    assert geo.lookup("10001", csv_path) == (40.7506, -73.9972)
    assert geo.lookup("10001-1234", csv_path) == (40.7506, -73.9972)
    assert geo.lookup("02134", csv_path) is not None  # leading zero kept
    assert geo.lookup("99999", csv_path) is None
    assert geo.lookup("abc", csv_path) is None
    assert geo.lookup("", csv_path) is None


def test_nearest_site(csv_path):
    locs = [
        {"city": "Los Angeles", "state": "CA", "lat": 34.05, "lon": -118.24},
        {"city": "Newark", "state": "NJ", "lat": 40.73, "lon": -74.17},
        {"city": "Unknown", "state": None, "lat": None, "lon": None},
    ]
    r = geo.nearest_site(locs, "10001", csv_path)
    assert r["city"] == "Newark" and r["distance_mi"] < 15


def test_nearest_site_none(csv_path):
    nogeo = [{"city": "x", "state": None, "lat": None, "lon": None}]
    assert geo.nearest_site(nogeo, "10001", csv_path) is None
    withgeo = [{"city": "x", "state": "NJ", "lat": 1.0, "lon": 1.0}]
    assert geo.nearest_site(withgeo, "99999", csv_path) is None


def test_real_centroids_file():
    geo.load_centroids.cache_clear()
    assert geo.lookup("10001") is not None
