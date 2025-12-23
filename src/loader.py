import pandas as pd
import numpy as np
from typing import Tuple, List

def load_spotify_data(filepath: str) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
    # load csv
    df = pd.read_csv(filepath, encoding='ISO-8859-1', index_col=0)
    
    # get genres
    if 'genre' in df.columns:
        genres = df['genre'].tolist()
    else:
        genres = df.iloc[:, 2].tolist()
    
    # keep numerics only
    num_data = df.select_dtypes(include=['float64', 'int64'])
    
    X = num_data.to_numpy()
    feats = num_data.columns.tolist()
    
    return X, genres, feats, df