import pandas as pd
import plotly.express as px
from functions import (
    infer_column_types, descriptive_statistics, analyze_numeric_columns, 
    analyze_categorical_columns, analyze_datetime_columns, correlation_analysis, 
    numeric_categorical_analysis, categorical_categorical_analysis
)

df = pd.DataFrame({
    "A": ['1', '2', '3', '4', '5'],
    "B": [10, 20, 30, 40, 50], 
    "C": [100, 200, 300, 400, 500]
})

# Create user_types dictionary mapping columns to their types
user_types = {
    "A": "Categorical",
    "B": "Numeric",
    "C": "Numeric"
}

# Call numeric_categorical_analysis with proper user_types dictionary
deneme = numeric_categorical_analysis(df, user_types)

print(deneme)
