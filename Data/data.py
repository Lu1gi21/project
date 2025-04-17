import pandas as pd
import numpy as np
import os

def read_vgsales_csv():
    """Read the vgsales.csv file and return its contents as a pandas DataFrame."""
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, 'vgsales.csv')
    df = pd.read_csv(file_path)
    return df

def clean_vgsales_data(df):
    """Clean and process the vgsales DataFrame for use in machine learning."""
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    
    for col in sales_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Year'] + sales_cols)
    df = df.reset_index(drop=True)
    return df

def load_clean_vgsales():
    """Read the vgsales.csv file, clean the data, and return the processed DataFrame."""
    df = read_vgsales_csv()
    df_clean = clean_vgsales_data(df)
    return df_clean

def get_numeric_vgsales_columns(df):
    """Return only the numeric columns relevant for modeling from the cleaned vgsales DataFrame."""
    numeric_cols = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    return df[numeric_cols]

def encode_categorical_column(df, column, method='label'):
    """
    Encode a categorical column using either label or one-hot encoding.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column to encode.
        method (str): 'label' or 'onehot'
    
    Returns:
        pandas.DataFrame: Updated DataFrame with encoded columns.
    """
    if method == 'label':
        unique_vals = sorted(df[column].dropna().unique())
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        df[column + '_encoded'] = df[column].map(val_to_int)
        return df
    elif method == 'onehot':
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        return df
    else:
        raise ValueError("Encoding method must be 'label' or 'onehot'")


