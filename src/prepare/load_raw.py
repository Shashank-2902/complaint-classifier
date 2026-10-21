"""Read the NHTSA flat files (zip, tab separated, no header)."""
import zipfile

import pandas as pd

from src import config


def read_raw_zip(path):
    """Read one NHTSA zip file and return only the columns we need, all as text."""
    frames = []
    with zipfile.ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as file:
            reader = pd.read_csv(
                file, sep="\t", header=None, usecols=list(config.RAW_COLUMNS), dtype=str,
                encoding="utf-8", encoding_errors="replace", quoting=3, chunksize=200_000, on_bad_lines="skip",
            )
            for chunk in reader:
                frames.append(chunk.rename(columns=config.RAW_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def read_raw_dir(directory=config.RAW_DIR):
    """Read every NHTSA zip file of this project from the directory."""
    frames = [read_raw_zip(directory / f"{name}.zip") for name in config.RAW_FILES]
    return pd.concat(frames, ignore_index=True)


def read_raw_tsv(path):
    """Read an unzipped flat file. Used for the small test fixture."""
    frame = pd.read_csv(path, sep="\t", header=None, usecols=list(config.RAW_COLUMNS), dtype=str,
                        encoding="utf-8", encoding_errors="replace", quoting=3)
    return frame.rename(columns=config.RAW_COLUMNS)
