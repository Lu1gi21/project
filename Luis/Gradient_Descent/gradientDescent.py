import sys
import os
from Data import load_clean_vgsales, get_numeric_vgsales_columns
data = load_clean_vgsales()

print(data.head())
print(data.columns)
numeric_data = get_numeric_vgsales_columns(data)

print(numeric_data.head())
print(numeric_data.columns)




