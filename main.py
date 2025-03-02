import streamlit as st
import pandas as pd
from functions import (
    infer_column_types, descriptive_statistics, analyze_numeric_columns, 
    analyze_categorical_columns, analyze_datetime_columns, correlation_analysis, 
    numeric_categorical_analysis, categorical_categorical_analysis
)

st.title("Data Analysis Platform")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, thousands=",")
    st.write("## Preview of Uploaded Data")
    st.dataframe(df.head())

    # Infer and verify column types
    inferred_types = infer_column_types(df)
    user_types = {}
    ordinal_columns = {}

    st.write("## Verify Column Types")
    for col, inferred_type in inferred_types.items():
        user_types[col] = st.selectbox(
            f"Select type for '{col}'",
            ["Numeric", "Categorical", "Datetime"],
            index=["Numeric", "Categorical", "Datetime"].index(inferred_type),
        )

        if user_types[col] == "Categorical":
            category_type = st.radio(f"Is '{col}' ordinal or nominal?", ["Nominal", "Ordinal"], key=col)
            if category_type == "Ordinal":
                unique_values = sorted(df[col].dropna().unique())
                ordinal_columns[col] = []
                for i, val in enumerate(unique_values, start=1):
                    order = st.number_input(
                        f"Order for '{val}' in '{col}'",
                        min_value=1, step=1, value=i,
                        key=f"order_{col}_{val}"
                    )
                    ordinal_columns[col].append((val, order))
                ordinal_columns[col].sort(key=lambda x: x[1])

    # Persist analysis selection
    if "selected_analysis" not in st.session_state:
        st.session_state.selected_analysis = None

    st.write("## Select Analysis")
    selected_analysis = st.selectbox("Choose an analysis type", ["Descriptive Statistics", "Discover Relationships"], key="analysis_type")

    if st.button("Proceed"):
        st.session_state.selected_analysis = selected_analysis

    # Descriptive Statistics Analysis
    if st.session_state.selected_analysis == "Descriptive Statistics":
        st.write("## Descriptive Statistics")
        show_numeric = st.checkbox("Show Numeric Analysis")
        show_categorical = st.checkbox("Show Categorical Analysis")
        show_datetime = st.checkbox("Show Datetime Analysis")

        if st.button("Calculate Statistics"):
            st.write("### Overall Statistics")
            desc_stats = descriptive_statistics(df)
            st.dataframe(desc_stats)

            numeric_cols = [col for col, col_type in user_types.items() if col_type == "Numeric"]
            categorical_cols = [col for col, col_type in user_types.items() if col_type == "Categorical"]
            datetime_cols = [col for col, col_type in user_types.items() if col_type == "Datetime"]

            if show_numeric and numeric_cols:
                numeric_results = analyze_numeric_columns(df, numeric_cols)
                for col, res in numeric_results.items():
                    st.write(f"#### Distribution of {col}")
                    st.plotly_chart(res["fig"])
                    st.write(f"Skewness: {res['skew']:.2f}")
                    st.write(f"Kurtosis: {res['kurtosis']:.2f}")

            if show_categorical and categorical_cols:
                categorical_results = analyze_categorical_columns(df, categorical_cols)
                for col, res in categorical_results.items():
                    st.write(f"#### Distribution of {col}")
                    st.plotly_chart(res["fig"])
                    st.dataframe(res["freq_table"])

            if show_datetime and datetime_cols:
                datetime_results = analyze_datetime_columns(df, datetime_cols)
                for col, res in datetime_results.items():
                    st.write(f"#### Analysis of {col}")
                    st.write(f"Time Range: {res['time_range']}")
                    st.write(f"Earliest Date: {res['earliest']}")
                    st.write(f"Latest Date: {res['latest']}")
                    st.plotly_chart(res["fig"])

    # Discover Relationships Analysis
    elif st.session_state.selected_analysis == "Discover Relationships":
        st.write("## Select Relationship Analysis")
        show_num_num = st.checkbox("Numeric - Numeric")
        show_num_cat = st.checkbox("Numeric - Categorical")
        show_cat_cat = st.checkbox("Categorical - Categorical")

        if st.button("Calculate Relationships"):
            if show_num_num:
                numeric_cols = [col for col, col_type in user_types.items() if col_type == "Numeric"]
                ordinal_cols = [col for col in ordinal_columns.keys()]
                correlation_cols = numeric_cols + ordinal_cols

                if len(correlation_cols) >= 2:
                    corr_fig, strong_corrs = correlation_analysis(df, correlation_cols, ordinal_columns)
                    st.pyplot(corr_fig)

                    if strong_corrs:
                        st.write("### Strong Correlations")
                        for col1, col2, corr in strong_corrs:
                            st.write(f"{col1} & {col2}: {corr:.2f}")
                            fig = px.scatter(df, x=col1, y=col2, labels={'x': col1, 'y': col2})
                            st.plotly_chart(fig)

            if show_num_cat:
                num_cat_results = numeric_categorical_analysis(df, user_types)
                for res in num_cat_results:
                    st.write(f"{res['cat_col']} - {res['num_col']}: {res['test']}, p-value = {res['p_value']:.4f}")
                    st.plotly_chart(res["fig"])

            if show_cat_cat:
                categorical_cols = [col for col, col_type in user_types.items() if col_type == "Categorical"]
                cat_cat_results = categorical_categorical_analysis(df, categorical_cols)
                for res in cat_cat_results:
                    st.write(f"{res['col1']} & {res['col2']}: p-value = {res['p_value']:.4f}")
                    st.plotly_chart(res["fig"])
