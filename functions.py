import pandas as pd
import plotly.express as px
import seaborn as sns
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def infer_column_types(df):
    inferred_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inferred_types[col] = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            inferred_types[col] = "Datetime"
        else:
            inferred_types[col] = "Categorical"
    return inferred_types

def descriptive_statistics(df):
    return df.describe(include='all')

def analyze_numeric_columns(df, numeric_cols):
    results = {}
    for col in numeric_cols:
        fig = px.histogram(df, x=col, marginal="box")
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        results[col] = {"fig": fig, "skew": skew, "kurtosis": kurt}
    return results

def analyze_categorical_columns(df, categorical_cols):
    results = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': col, 'y': 'Count'})
        freq_table = pd.DataFrame({
            'Count': value_counts,
            'Percentage': (value_counts / len(df) * 100).round(2)
        })
        results[col] = {"fig": fig, "freq_table": freq_table}
    return results

def analyze_datetime_columns(df, datetime_cols):
    results = {}
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
        time_range = df[col].max() - df[col].min()
        earliest_date = df[col].min()
        latest_date = df[col].max()
        fig = px.line(df, x=col, title=f"Timeline of {col}")
        results[col] = {"fig": fig, "time_range": time_range, "earliest": earliest_date, "latest": latest_date}
    return results

def correlation_analysis(df, correlation_cols, ordinal_columns):
    df_corr = df[correlation_cols].copy()
    for col in ordinal_columns.keys():
        order_map = {val: order for val, order in ordinal_columns[col]}
        df_corr[col] = df[col].map(order_map)
    
    corr_matrix = df_corr.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    
    strong_corrs = [(col1, col2, corr_matrix.loc[col1, col2]) for col1 in correlation_cols for col2 in correlation_cols if col1 < col2 and abs(corr_matrix.loc[col1, col2]) > 0.6]
    strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
    
    return fig, strong_corrs

def numeric_categorical_analysis(df, user_types):
    results = []
    for cat_col in [col for col, col_type in user_types.items() if col_type == "Categorical"]:
        for num_col in [col for col, col_type in user_types.items() if col_type in ["Numeric", "Ordinal"]]:
            fig = px.box(df, x=cat_col, y=num_col)
            
            groups = df.groupby(cat_col)[num_col]
            normal_distribution = all(stats.normaltest(group)[1] >= 0.01 for name, group in groups)
            
            if normal_distribution:
                f_stat, p_value = stats.f_oneway(*[group for name, group in groups])
                test_type = "ANOVA"
                metric = groups.mean().mean()
            else:
                h_stat, p_value = stats.kruskal(*[group for name, group in groups])
                test_type = "Kruskal-Wallis"
                metric = groups.median().median()
            
            results.append({
                'cat_col': cat_col,
                'num_col': num_col,
                'test': test_type,
                'p_value': p_value,
                'metric': metric,
                'fig': fig
            })
    return results

def categorical_categorical_analysis(df, categorical_cols):
    results = []
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
            fig = px.imshow(contingency_table, text_auto=True, color_continuous_scale="blues")
            results.append({
                'col1': col1,
                'col2': col2,
                'p_value': p,
                'fig': fig
            })
    return results
