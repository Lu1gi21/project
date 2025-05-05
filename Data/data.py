import pandas as pd

def read_vgsales_csv():
    return pd.read_csv('Data/vgsales.csv')

def clean_vgsales_data(df):
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in sales_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Year'] + sales_cols)

    return df

def load_clean_vgsales():
    df = read_vgsales_csv()
    df_clean = clean_vgsales_data(df)
    return df_clean

def get_numeric_vgsales_columns(df):
    numeric_cols = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    return df[numeric_cols]


