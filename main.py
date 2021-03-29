import numpy as np
import pandas as pd


def _filter_columns(iterator: pd.DataFrame, string_: str):
    if isinstance(iterator, pd.DataFrame):
        iterator = iterator.columns
    columns_ = []
    for column in iterator:
        if column.startswith(string_):
            columns_.append(column)
    return columns_


def combine_predictions(row: pd.DataFrame, string: str = 'preds', cv: bool = False):
    x = np.concatenate([row[column] for column in _filter_columns(row, string)])
    return np.unique(x) if cv else ' '.join(np.unique(x))
