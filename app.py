import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loading import load_sales, load_reviews, load_churn
from src.eda import plot_top_products, plot_monthly_sales, plot_revenue_by_category
from src.sentiment import analyze_reviews, sentiment_distribution
from src.churn import train_churn_model, save_model, load_model, confusion_matrix_fig, roc_curve_fig, FEATURES_NUM, FEATURES_CAT, TARGET

st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")

DATA_DIR = Path('data')
MODEL_PATH = Path('models/churn_model.pkl')

st.sidebar.title("Data & Settings")

# Sales data
st.sidebar.subheader("Sales Data")
sales_file = st.sidebar.file_uploader("Upload sales CSV", type=['csv'], key='sales')
if sales_file is None:
    sales_path = DATA_DIR / 'sales_sample.csv'
else:
    sales_path = sales_file

# Reviews data
st.sidebar.subheader("Reviews Data")
reviews_file = st.sidebar.file_uploader("Upload reviews CSV", type=['csv'], key='reviews')
if reviews_file is None:
    reviews_path = DATA_DIR / 'reviews_sample.csv'
else:
    reviews_path = reviews_file

# Churn data
st.sidebar.subheader("Churn Data")
churn_file = st.sidebar.file_uploader("Upload churn CSV", type=['csv'], key='churn')
if churn_file is None:
    churn_path = DATA_DIR / 'churn_sample.csv'
else:
    churn_path = churn_file

st.title("📊 Customer Insights Dashboard")
st.caption("Sales Analytics • Customer Emotions (NLP) • Churn Prediction (ML)")

tab1, tab2, tab3 = st.tabs(["Sales Analysis", "Customer Emotions", "Churn Prediction"])

with tab1:
    st.header("Sales Analysis")
    try:
        # 1) Veriyi yükle
        df_sales = load_sales(sales_path)

        # 2) Filtreler
        date_range = st.date_input(
            "Date range",
            value=(df_sales['order_date'].min(), df_sales['order_date'].max())
        )
        top_n = st.slider("Top N products", min_value=3, max_value=20, value=5, step=1)

        # 3) Tarih filtresini uygula
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_sales = df_sales[(df_sales['order_date'] >= start) & (df_sales['order_date'] <= end)]

        st.write(f"Loaded **{len(df_sales)}** sales rows.")

        # 4) Grafikleri çiz
        c1, c2 = st.columns(2)
        with c1:
            fig1 = plot_top_products(df_sales, top_n=top_n)
            st.pyplot(fig1, clear_figure=True)
        with c2:
            fig2 = plot_monthly_sales(df_sales)
            st.pyplot(fig2, clear_figure=True)

        st.markdown('---')
        fig3 = plot_revenue_by_category(df_sales, top_n=5)
        st.pyplot(fig3, clear_figure=True)

        # 5) BI'ye uygun CSV indirme (revenue + order_month ekli)
        st.subheader("Export")
        df_export = df_sales.copy()
        df_export['revenue'] = df_export['quantity'] * df_export['unit_price']
        df_export['order_month'] = df_export['order_date'].dt.to_period('M').astype(str)
        st.download_button(
            "Download Sales (enriched) CSV",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name="sales_enriched.csv",
            mime="text/csv"
        )

        # 6) Ad-hoc SQL çalışma alanı (filtrelenmiş satış verisi üzerinde)
        with st.expander("Run ad-hoc SQL on filtered sales"):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            df_sql = df_sales.copy()
            df_sql.to_sql("sales", conn, index=False, if_exists="replace")

            default_query = """
            SELECT product_category,
                   SUM(quantity * unit_price) AS revenue,
                   COUNT(*) AS rows_cnt
            FROM sales
            GROUP BY product_category
            ORDER BY revenue DESC
            LIMIT 5;
            """
            query = st.text_area("SQL", value=default_query, height=140)
            if st.button("Run SQL"):
                try:
                    res = pd.read_sql_query(query, conn)
                    st.dataframe(res, use_container_width=True)
                    st.download_button(
                        "Download SQL result CSV",
                        data=res.to_csv(index=False).encode("utf-8"),
                        file_name="sql_result.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"SQL error: {e}")
            conn.close()

        st.info("Tip: Replace `data/sales_sample.csv` with your real dataset or upload from the sidebar.")

    except Exception as e:
        st.error(f"Failed to analyze sales data: {e}")


with tab2:
    st.header("Customer Emotions (Sentiment)")

    try:
        # 1) Veriyi yükle
        df_reviews = load_reviews(reviews_path)
        st.write(f"Loaded **{len(df_reviews)}** reviews.")

        # 2) Kaç yorum işlenecek? (zayıf laptoplar için güvenli)
        limit = st.slider("Number of reviews to process", min_value=10, max_value=500, value=min(200, len(df_reviews)), step=10)

        # 3) Sentiment analizi (HF varsa otomatik; yoksa hızlı fallback)
        df_sent = analyze_reviews(df_reviews, limit=limit)

        # 4) Dağılım grafiği
        dist = sentiment_distribution(df_sent)
        fig, ax = plt.subplots()
        dist.sort_index().plot(kind='bar', ax=ax)
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Share')
        st.pyplot(fig, clear_figure=True)

        # 5) Örnek satırlar tablosu
        st.dataframe(df_sent.head(10), use_container_width=True)

        # 6) İndirme: Sentiment’li CSV
        st.subheader("Export")
        st.download_button(
            "Download Reviews with Sentiment CSV",
            data=df_sent.to_csv(index=False).encode('utf-8'),
            file_name="reviews_with_sentiment.csv",
            mime="text/csv"
        )

        # 7) Bilgi notu (opsiyonel ama faydalı)
        st.caption("If Transformers is installed, model-based sentiment runs; otherwise a lightweight rule-based fallback is used.")

    except Exception as e:
        st.error(f"Failed to analyze reviews: {e}")


with tab3:
    st.header("Churn Prediction")

    try:
        # 1) Veriyi yükle
        df_churn = load_churn(churn_path)
        st.write(f"Loaded **{len(df_churn)}** churn rows.")

        # 2) Zorunlu kolonları kontrol et
        REQUIRED_NUM = ['tenure_months','monthly_charges','total_charges']
        REQUIRED_CAT = ['contract_type','has_complaints']
        TARGET = 'churn_label'
        missing = [c for c in REQUIRED_NUM + REQUIRED_CAT if c not in df_churn.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # 3) Tipleri düzelt & NaN temizle
        dfc = df_churn.copy()
        dfc['contract_type'] = dfc['contract_type'].astype(str).str.strip().str.lower()
        for col in REQUIRED_NUM + ['has_complaints', TARGET]:
            if col in dfc.columns:
                dfc[col] = pd.to_numeric(dfc[col], errors='coerce')

        # Feature tarafında NaN varsa at
        dfc = dfc.dropna(subset=REQUIRED_NUM + REQUIRED_CAT)

        # 4) Ayarlar
        retrain = st.checkbox("Retrain model on loaded churn dataset", value=False)
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)

        # 5) Modeli yükle / eğit (bozuk model dosyası tespit edilirse otomatik kurtar)
        from pathlib import Path
        MODEL_PATH = Path('models/churn_model.pkl')
        pipe = None

        if retrain:
            with st.spinner("Training model..."):
                pipe, metrics = train_churn_model(dfc)
                save_model(pipe, MODEL_PATH)
                st.success(f"Model trained. AUC = {metrics.get('auc', float('nan')):.3f}")
        else:
            if MODEL_PATH.exists():
                try:
                    pipe = load_model(MODEL_PATH)
                    st.success("Loaded existing churn_model.pkl")
                except Exception as e:
                    st.warning(f"Existing model couldn't be loaded ({e}). Retraining now...")
                    pipe, metrics = train_churn_model(dfc)
                    save_model(pipe, MODEL_PATH)
                    st.success(f"Model re-trained. AUC = {metrics.get('auc', float('nan')):.3f}")
            else:
                pipe, metrics = train_churn_model(dfc)
                save_model(pipe, MODEL_PATH)
                st.success(f"Model trained (no previous model). AUC = {metrics.get('auc', float('nan')):.3f}")

        # 6) Tahminler
        X = dfc[['tenure_months','monthly_charges','total_charges','contract_type','has_complaints']]
        y = None
        if TARGET in dfc.columns:
            try:
                y = dfc[TARGET].astype(int)
            except Exception:
                y = pd.to_numeric(dfc[TARGET], errors='coerce').fillna(0).astype(int)

        y_prob = pipe.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # 7) Grafikler (y varsa CM/ROC göster)
        c1, c2 = st.columns(2)
        if y is not None and y.notna().any():
            with c1:
                fig_cm = confusion_matrix_fig(y, y_pred)
                st.pyplot(fig_cm, clear_figure=True)
            with c2:
                fig_roc = roc_curve_fig(y, y_prob)
                st.pyplot(fig_roc, clear_figure=True)

            from sklearn.metrics import classification_report
            st.text("Classification Report:")
            st.code(classification_report(y, y_pred))
        else:
            st.info("No ground-truth labels found; showing probabilities only.")

        # 8) İndirilebilir sonuç
        out_df = dfc.copy()
        out_df['churn_probability'] = y_prob
        out_df['predicted_label'] = y_pred
        st.download_button(
            "Download Churn Predictions CSV",
            data=out_df.to_csv(index=False).encode('utf-8'),
            file_name="churn_with_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        # Tam hata/traceback göster ki teşhis edebilelim
        st.exception(e)
