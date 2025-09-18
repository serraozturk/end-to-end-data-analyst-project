import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    for col in ['quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['unit_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ['review_text']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_churn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ['tenure_months','monthly_charges','total_charges','has_complaints','churn_label']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'contract_type' in df.columns:
        df['contract_type'] = df['contract_type'].astype(str).str.strip().str.lower()
    return df
