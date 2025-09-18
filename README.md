# Customer Insights Dashboard (End-to-End: Analyst + NLP + ML)

An end-to-end portfolio project that combines **Sales Analytics**, **Customer Sentiment (NLP)**, and **Churn Prediction (ML)** into a single interactive dashboard.

This starter includes:
- Sample datasets under `data/`
- A Streamlit app (`app.py`) with 3 tabs: Sales / Emotions / Churn
- Helper modules in `src/`
- A pre-trained (toy) churn model under `models/` (generated from the sample data)
- Minimal & optional requirements files

## 0) Setup (Windows/Mac/Linux)

```bash
# create & activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# install minimal requirements
pip install -r requirements-min.txt

# (Optional) for NLP sentiment using Transformers (CPU inference)
pip install -r requirements-nlp.txt
```

## 1) Run the app

```bash
streamlit run app.py
```

- By default it loads the small sample CSV files under `data/`.
- You can upload your own CSVs from the sidebar.

## 2) Project Structure

```
customer_insights_dashboard/
├─ app.py
├─ README.md
├─ requirements-min.txt
├─ requirements-nlp.txt
├─ data/
│  ├─ sales_sample.csv
│  ├─ reviews_sample.csv
│  └─ churn_sample.csv
├─ models/
│  └─ churn_model.pkl         # a toy LogisticRegression model
├─ src/
│  ├─ data_loading.py
│  ├─ eda.py
│  ├─ sentiment.py
│  └─ churn.py
├─ notebooks/                 # (optional) your EDA notebooks
└─ reports/                   # (optional) export plots/reports here
```

## 3) Tabs Overview

### Sales (Data Analyst)
- Top products, monthly sales trend, revenue by category/city/country

### Emotions (NLP)
- Sentiment analysis of customer reviews
- Uses HuggingFace Transformers pipeline if available; falls back to a tiny rule-based approach otherwise

### Churn (ML)
- Logistic Regression on sample features
- Shows confusion matrix and ROC
- You can (re)train from the UI using your own `churn.csv`

## 4) Datasets

- Replace the files in `data/` with your own datasets or upload via the UI.
- Expected columns (you can adapt in code if needed):

**sales.csv**
order_id,order_date,product_category,product_name,quantity,unit_price,customer_id,city,country

**reviews.csv**
review_id,product_name,review_text

**churn.csv**
customer_id,tenure_months,monthly_charges,total_charges,contract_type,has_complaints,churn_label

## 5) Notes

- NLP transformers may download a model on first use (internet needed). On CPU it typically runs fine for small batches.
- If your laptop is slow, process a smaller subset of reviews (slider in the UI).
- The included churn model is a toy; for real datasets, use the retrain option.
