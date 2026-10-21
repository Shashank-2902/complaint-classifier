"""Download the NHTSA complaint files (about 190 MB) into data/raw/.

    python -m src.prepare.download

Source: https://www.nhtsa.gov/nhtsa-datasets-and-apis (section "Complaints", five-year sets).
"""
import urllib.request

from src import config


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.RAW_FILES:
        target = config.RAW_DIR / f"{name}.zip"
        if target.exists():
            print(f"{name}: already downloaded")
            continue
        print(f"{name}: downloading ...", flush=True)
        urllib.request.urlretrieve(config.RAW_URL.format(name=name), target)
        print(f"{name}: {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
