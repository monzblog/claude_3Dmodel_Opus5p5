"""OSM と標高タイルだけを取得して data/c1_data.zip にまとめる（Blender は不要）。

    pip install requests numpy pyproj
    python -m c1gen.fetch
"""
import os
import zipfile

from . import dem, osm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    os.makedirs(DATA, exist_ok=True)
    print("OSM を取得中…")
    osm.fetch(os.path.join(DATA, "osm_c1.json"))
    print("標高タイルを取得中…")
    dem.fetch(os.path.join(DATA, "dem"))
    out = os.path.join(DATA, "c1_data.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(DATA, "osm_c1.json"), "osm_c1.json")
        for base, _, files in os.walk(os.path.join(DATA, "dem")):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, DATA))
    print(f"できました: {out}")


if __name__ == "__main__":
    main()
