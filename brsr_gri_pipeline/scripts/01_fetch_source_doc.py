"""
Downloads the official GRI-BRSR linkage document.

NOTE: run this on your own machine / CI runner with normal internet access.
It will NOT work in a network-restricted sandbox (globalreporting.org needs
to be reachable).

Usage:
    python scripts/01_fetch_source_doc.py
"""

import pathlib
import requests

URL = "https://www.globalreporting.org/media/ioqnxtmx/sebi_brsb_gri_linkage_doc.pdf"
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "gri_brsr_linkage_doc.pdf"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    OUT_PATH.write_bytes(resp.content)
    print(f"Saved to {OUT_PATH} ({len(resp.content) / 1024:.1f} KB)")
    print("\nIf this fails (403/timeout), download manually from:")
    print(f"  {URL}")
    print(f"and place it at:\n  {OUT_PATH}")


if __name__ == "__main__":
    main()
