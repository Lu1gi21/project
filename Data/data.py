import pandas as pd
import os

def read_vgsales_csv():
    """Read the vgsales.csv file and return its contents as a pandas DataFrame.

    Args:
        filepath (str): The path to the vgsales.csv file.

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data.
    """
    dir_path = os.path.dirname(__file__)  # this gets the folder containing data.py
    file_path = os.path.join(dir_path, 'vgsales.csv')
    return pd.read_csv(file_path)

def clean_vgsales_data(df):
    """Clean and process the vgsales DataFrame for use in machine learning.

    This function drops rows with missing values in key columns and converts columns to appropriate types.
    All columns are retained in the returned DataFrame.

    Args:
        df (pandas.DataFrame): Raw vgsales DataFrame.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with all columns retained.
    """
    # Convert 'Year' and sales columns to numeric, coercing errors to NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in sales_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any NaN values in these columns
    df = df.dropna(subset=['Year'] + sales_cols)

    return df

def load_clean_vgsales():
    """Read the vgsales.csv file, clean the data, and return the processed DataFrame.

    Args:
        filepath (str): The path to the vgsales.csv file.

    Returns:
        pandas.DataFrame: Cleaned DataFrame ready for modeling.
    """
    df = read_vgsales_csv()
    df_clean = clean_vgsales_data(df)
    return df_clean

def get_numeric_vgsales_columns(df):
    """Return only the numeric columns relevant for modeling from the cleaned vgsales DataFrame.

    Args:
        df (pandas.DataFrame): Cleaned vgsales DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing only the numeric columns for modeling.
    """
    numeric_cols = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    return df[numeric_cols]


