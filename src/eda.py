import pandas as pd
import matplotlib.pyplot as plt

def plot_top_products(df: pd.DataFrame, top_n: int = 5):
    df = df.copy()
    df['revenue'] = df['quantity'] * df['unit_price']
    by_product = df.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots()
    by_product.plot(kind='bar', ax=ax)
    ax.set_title(f'Top {top_n} Products by Revenue')
    ax.set_xlabel('Product')
    ax.set_ylabel('Revenue')
    fig.tight_layout()
    return fig

def plot_monthly_sales(df: pd.DataFrame):
    df = df.copy()
    df['revenue'] = df['quantity'] * df['unit_price']
    monthly = df.set_index('order_date').resample('M')['revenue'].sum()
    fig, ax = plt.subplots()
    monthly.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Monthly Sales Revenue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')
    fig.tight_layout()
    return fig

def plot_revenue_by_category(df: pd.DataFrame, top_n: int = 5):
    df = df.copy()
    df['revenue'] = df['quantity'] * df['unit_price']
    by_cat = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots()
    by_cat.plot(kind='bar', ax=ax)
    ax.set_title(f'Revenue by Category (Top {top_n})')
    ax.set_xlabel('Category')
    ax.set_ylabel('Revenue')
    fig.tight_layout()
    return fig
