from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

FEATURES_NUM = ['tenure_months','monthly_charges','total_charges']
FEATURES_CAT = ['contract_type','has_complaints']
TARGET = 'churn_label'

def build_pipeline() -> Pipeline:
    num_tf = Pipeline(steps=[('scale', StandardScaler())])
    cat_tf = OneHotEncoder(handle_unknown='ignore')
    pre = ColumnTransformer([
        ('num', num_tf, FEATURES_NUM),
        ('cat', cat_tf, FEATURES_CAT),
    ])
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([('prep', pre), ('clf', model)])
    return pipe

def train_churn_model(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    df = df.dropna(subset=FEATURES_NUM + FEATURES_CAT + [TARGET]).copy()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET].astype(int)
    pipe = build_pipeline()
    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y, y_pred, output_dict=True)
    auc = roc_auc_score(y, y_prob)
    metrics = {'auc': float(auc), 'report': report}
    return pipe, metrics

def save_model(pipe: Pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)

def load_model(path: str) -> Pipeline:
    return joblib.load(path)

def confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def roc_curve_fig(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    fig.tight_layout()
    return fig
