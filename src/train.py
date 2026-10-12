"""Train, cross-validate and evaluate churn models.

Usage:  python -m src.train
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (RocCurveDisplay, classification_report, roc_auc_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import data, eda
from src.features import build_preprocessor

SEED = 42


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=2.8, random_state=SEED,
                                 eval_metric="logloss"),
    }


def main():
    df = data.load()
    eda.run(df)
    X, y = df.drop(columns="Churn"), df["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, clf in models().items():
        pipe = Pipeline([("prep", build_preprocessor(data.NUMERIC, data.categorical(X))), ("clf", clf)])
        cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p >= 0.5, average="binary")
        results[name] = dict(cv_auc_mean=round(cvs.mean(), 4), cv_auc_std=round(cvs.std(), 4),
                             test_auc=round(roc_auc_score(yte, p), 4), precision=round(pr, 3),
                             recall=round(rc, 3), f1=round(f1, 3))
        fitted[name] = pipe
        RocCurveDisplay.from_predictions(yte, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("ROC curves (hold-out test set)")
    fig.tight_layout(); fig.savefig("images/roc_curves.png", dpi=120); plt.close(fig)

    res = pd.DataFrame(results).T
    print(res.to_string())
    res.to_csv("reports/model_comparison.csv")

    # Logistic regression coefficients (interpretable drivers)
    lr = fitted["Logistic Regression"]
    coef = pd.Series(lr["clf"].coef_[0], index=lr["prep"].get_feature_names_out()).sort_values()
    top = pd.concat([coef.head(8), coef.tail(8)])
    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot.barh(ax=ax, color=["#55A868" if v < 0 else "#C44E52" for v in top])
    ax.set_title("Logistic regression: churn drivers (log-odds)")
    fig.tight_layout(); fig.savefig("images/lr_coefficients.png", dpi=120); plt.close(fig)

    # SHAP for XGBoost
    xgb = fitted["XGBoost"]
    Xt = pd.DataFrame(xgb["prep"].transform(Xte), columns=xgb["prep"].get_feature_names_out())
    sv = shap.TreeExplainer(xgb["clf"]).shap_values(Xt)
    shap.summary_plot(sv, Xt, max_display=12, show=False)
    plt.title("XGBoost – SHAP feature impact"); plt.tight_layout()
    plt.savefig("images/shap_summary.png", dpi=120); plt.close("all")
    imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)

    # Business view: capture of churners when targeting top-risk deciles
    p = fitted["Logistic Regression"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p); ys = yte.values[order]
    top20 = ys[: int(0.2 * len(ys))].sum() / ys.sum()
    top30 = ys[: int(0.3 * len(ys))].sum() / ys.sum()
    print(classification_report(yte, p >= 0.5))
    summary = dict(models=results, base_churn_rate=round(y.mean(), 4), n_rows=len(df),
                   top_shap=imp.head(8).round(4).to_dict(),
                   lr_top20pct_captures=round(top20, 3), lr_top30pct_captures=round(top30, 3),
                   churn_by_contract=df.groupby("Contract")["Churn"].mean().round(3).to_dict())
    json.dump(summary, open("reports/metrics.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
