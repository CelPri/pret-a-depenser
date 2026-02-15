import pandas as pd
import numpy as np


def dummy_preprocess(df: pd.DataFrame):
   
    return df.fillna(0)


def test_preprocessing_no_crash_and_shape_preserved():
    df = pd.DataFrame({
        "A": [1, 2, np.nan],
        "B": [5, np.nan, 7]
    })

    processed = dummy_preprocess(df)

    # même nombre de lignes
    assert processed.shape[0] == df.shape[0]

    # pas de NaN après traitement
    assert not processed.isna().any().any()