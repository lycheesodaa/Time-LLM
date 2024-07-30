import pandas as pd
import numpy as np
import os

cleaned_dir = 'cleaned/'
cleaned = os.listdir(cleaned_dir)
top50 = os.listdir()
for file in top50:
    if file in cleaned or not file.endswith('.csv'):
        continue
    df = pd.read_csv(file)
    print(f'Number of rows in {file}: {len(df)}')
    print(f'Number of columns in {file}: {len(df.columns)}')

    # drop columns with holes between values
    mask = df.notna() & df.shift(-1).notna() & df.shift().isna() & df.shift(-2).isna()
    columns_to_drop = mask.any()
    df = df.drop(columns=df.columns[columns_to_drop])

    # drop columns with more than half null values
    null_percent = df.isnull().mean()
    threshold = 0.4  # 50%
    cols_to_drop = null_percent[null_percent > threshold].index
    df = df.drop(cols_to_drop, axis=1)

    # drop rows at the beginning and end (ideally)
    df_cleaned = df.dropna()

    print(f'Number of rows remaining: {len(df_cleaned)}')
    print(f'Number of columns remaining: {len(df_cleaned.columns)}')

    if len(df_cleaned) < 10:
        raise Exception('Too few rows')
    df_cleaned.to_csv(cleaned_dir + file, index=False)