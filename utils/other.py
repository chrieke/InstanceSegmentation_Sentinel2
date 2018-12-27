# other.py

from typing import Union, Any, Callable, Dict
from pathlib import Path
import pickle

import json


def new_pickle(out_fp: Path, data):
    """(Over)write data to new pickle file."""
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {out_fp.name}')


def load_pickle(in_fp: Path):
    print(f'Loading from pickle file... {in_fp.name}')
    with open(in_fp, "rb") as f:
        return pickle.load(f)


def new_json(out_fp: Path, data):
    """(Over)write data to new pickle file."""
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as f:
        json.dump(data, f, indent=4)
    print(f'Writing new json file... {out_fp.name}')


def load_json(in_fp: Path):
    print(f'Loading from json file... {in_fp.name}')
    with open(in_fp, "r") as f:
        return json.load(f)


def load_or_new_save(fp: Path,
                     default_data: Union[Callable, Any],
                     callable_args: Dict=None,
                     file_format: str='pickle'
                     ) -> Any:
    """Write data to new pickle/json file or load pickle/json if that file already exists.

    Example:
        df = cgeo.other.load_or_new_save(fp=Path('output\preprocessed_marker_small.pkl'),
                                         default_data=preprocess_vector,
                                         callable_args={'infp': fp_fields, 'meta': meta})
    Args:
        fp: in/output pickle/json file fp.
        file_format: Either 'pickle' or 'json'.
        default_data: Data that is written to a pickle/json file if the pickle/json does not already exist.
            When giving a function, do not call the function, only give the function
            object name. Function arguments can be provided via callable_args.
        callable_args: args for additional function arguments when default_data is a callable function.

    Returns:
        Contents of the loaded or newly created pickle/json file.
    """
    if not fp.exists():
        if not callable(default_data):
            data = default_data
        else:
            if callable_args is None:
                data = default_data()
            else:
                data = default_data(**callable_args)
        if file_format == 'pickle':
            new_pickle(out_fp=fp, data=data)
        elif file_format == 'json':
            new_json(out_fp=fp, data=data)
    else:
        if file_format == 'pickle':
            data = load_pickle(fp)
        elif file_format == 'json':
            data = load_json(fp)
    return data
