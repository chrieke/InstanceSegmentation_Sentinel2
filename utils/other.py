# other.py

from pathlib import Path
import pickle

import json


def new_pickle(outpath: Path, data):
    """(Over)write data to new pickle file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {outpath.name}')


def load_pickle(inpath: Path):
    print(f'Loading from existing pickle file... {inpath.name}')
    with open(inpath, "rb") as f:
        return pickle.load(f)


def new_json(outpath: Path, data):
    """(Over)write data to new json file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=4)
    print(f'Writing new json file... {outpath.name}')


def load_json(inpath: Path):
    print(f'Loading from existing json file... {inpath.name}')
    with open(inpath, "r") as f:
        return json.load(f)
