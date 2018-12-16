# other.py

from typing import Union, Any, Callable, Dict
from pathlib import Path
import pickle

import json


def new_save(out_path: Path, data, file_format: str='pickle'):
    """(Over)write data to new pickle/json file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == 'pickle':
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
    elif file_format == 'json':
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
    print(f'Writing new {file_format} file... {out_path.name}')


def load_saved(in_path: Path, file_format: str='pickle'):
    """Load saved pickle/json file."""

    if file_format == 'pickle':
        with open(in_path, "rb") as f:
            data = pickle.load(f)
    elif file_format == 'json':
        with open(in_path, "r") as f:
            data = json.load(f)
    print(f'Loading from {file_format} file... {in_path.name}')
    return data


def load_or_new_save(path: Path,
                     default_data: Union[Callable, Any],
                     callable_args: Dict=None,
                     file_format: str='pickle'
                     ) -> Any:
    """Write data to new pickle/json file or load pickle/json if that file already exists.

    Example:
        df = utils.other.load_or_new_save(path=Path('output\preprocessed_marker_small.pkl'),
                                         default_data=preprocess_vector,
                                         callable_args={'inpath': fp_fields, 'meta': meta})
    Args:
        path: in/output pickle/json file path.
        file_format: Either 'pickle' or 'json'.
        default_data: Data that is written to a pickle/json file if the pickle/json does not already exist.
            When giving a function, do not call the function, only give the function
            object name. Function arguments can be provided via callable_args.
        callable_args: args for additional function arguments when default_data is a callable function.

    Returns:
        Contents of the loaded or newly created pickle/json file.
    """
    try:
        if file_format == 'pickle':
            data = load_saved(path, file_format=file_format)
        elif file_format == 'json':
            data = load_saved(path, file_format=file_format)
    except (FileNotFoundError, OSError, IOError, EOFError):
        if not callable(default_data):
            data = default_data
        else:
            if callable_args is None:
                data = default_data()
            else:
                data = default_data(**callable_args)
        if file_format == 'pickle':
            new_save(out_path=path, data=data, file_format=file_format)
        elif file_format == 'json':
            new_save(out_path=path, data=data, file_format=file_format)

    return data
