from typing import List
import pandas as pd

def _simple_rule_based_sentiment(text: str) -> str:
    text = text.lower()
    positives = ['good','great','love','amazing','happy','nice','soft','perfect','better']
    negatives = ['bad','poor','cheap','worse','worst','awful','hate','noisy','awkward','dies']
    p = sum(w in text for w in positives)
    n = sum(w in text for w in negatives)
    if p > n:
        return 'POSITIVE'
    if n > p:
        return 'NEGATIVE'
    return 'NEUTRAL'

def hf_sentiment_or_fallback(texts: List[str]) -> List[str]:
    try:
        from transformers import pipeline
        clf = pipeline("sentiment-analysis")
        out = clf(texts)
        labels = []
        for item in out:
            label = str(item.get('label','')).upper()
            if 'POS' in label:
                labels.append('POSITIVE')
            elif 'NEG' in label:
                labels.append('NEGATIVE')
            else:
                labels.append('NEUTRAL')
        return labels
    except Exception:
        return [_simple_rule_based_sentiment(t) for t in texts]

def analyze_reviews(df_reviews: pd.DataFrame, text_col: str = 'review_text', limit: int = 200) -> pd.DataFrame:
    df = df_reviews.copy()
    df = df.head(limit)
    labels = hf_sentiment_or_fallback(df[text_col].fillna('').astype(str).tolist())
    df['sentiment'] = labels
    return df

def sentiment_distribution(df_with_sentiment: pd.DataFrame) -> pd.Series:
    return df_with_sentiment['sentiment'].value_counts(normalize=True).sort_index()
