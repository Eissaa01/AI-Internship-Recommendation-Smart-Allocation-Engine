"""
model.py — ML model training, evaluation, and prediction pipeline
Uses Logistic Regression with engineered features.
Run directly:  python model.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import FEATURE_NAMES, build_feature_vector, explain_prediction

MODEL_PATH  = "model/internship_model.pkl"
SCALER_PATH = "model/scaler.pkl"
META_PATH   = "model/model_meta.json"


# Training 

def train(data_path="data/training_data.csv",
          model_type="logistic",
          save=True):
    """
    Load CSV → extract features → train model → evaluate → (optionally) save.
    Returns (pipeline, metadata_dict)
    """
    print(f"\n📊 Loading training data from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"   {len(df)} rows  |  {df['label'].mean():.1%} positive rate")

    # Features
    feature_cols = FEATURE_NAMES
    X = df[feature_cols].values.astype(float)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # pipeline
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=500, class_weight="balanced",
                                 solver="lbfgs", C=1.0, random_state=42)
        print("\n🔧 Training Logistic Regression …")
    else:
        clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                     random_state=42)
        print("\n🔧 Training Decision Tree …")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])
    pipeline.fit(X_train, y_train)

    # Eval
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_auc  = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc").mean()

    print(f"\n📈 Evaluation Results:")
    print(f"   Accuracy        : {acc:.3f}")
    print(f"   ROC-AUC (test)  : {auc:.3f}")
    print(f"   ROC-AUC (5-fold): {cv_auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # coefficients 
    if model_type == "logistic":
        raw_coefs = clf.coef_[0].tolist()
    else:
        raw_coefs = clf.feature_importances_.tolist()

    coef_dict = dict(zip(FEATURE_NAMES, raw_coefs))

    meta = {
        "model_type":   model_type,
        "accuracy":     round(acc, 4),
        "roc_auc":      round(auc, 4),
        "cv_auc":       round(cv_auc, 4),
        "feature_names": FEATURE_NAMES,
        "coef_dict":    {k: round(v, 4) for k, v in coef_dict.items()},
        "trained_on":   len(X_train),
    }

    if save:
        os.makedirs("model", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n✅ Model saved → {MODEL_PATH}")
        print(f"   Meta  saved → {META_PATH}")

    return pipeline, meta


# Loading 

_pipeline_cache = None
_meta_cache     = None

def load_model():
    global _pipeline_cache, _meta_cache
    if _pipeline_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python model.py"
            )
        with open(MODEL_PATH, "rb") as f:
            _pipeline_cache = pickle.load(f)
        with open(META_PATH, "r") as f:
            _meta_cache = json.load(f)
    return _pipeline_cache, _meta_cache


# Prediction 
def recommend(user_profile: dict,
              internship_list: list,
              top_n: int = 5) -> list:
    """
    For a given user profile, score every available internship,
    rank by probability, and return the top_n recommendations.

    Each result dict:
      {internship_id, title, sector, location, score,
       score_pct, features, explanation, stipend, duration_weeks}
    """
    pipeline, meta = load_model()
    coef_dict      = meta["coef_dict"]
    results        = []
    """ Hello """
    for intern in internship_list:
        fv_dict = build_feature_vector(user_profile, intern)
        fv_arr  = np.array([[fv_dict[k] for k in FEATURE_NAMES]])

        proba   = pipeline.predict_proba(fv_arr)[0][1]   # P(match=1)
        explanation = explain_prediction(fv_dict, coef_dict)

        results.append({
            "internship_id":  intern["internship_id"],
            "title":          intern.get("title", intern["internship_id"]),
            "sector":         intern.get("sector", intern.get("internship_sector", "")),
            "location":       intern.get("location", intern.get("internship_location", "")),
            "required_skills": intern.get("required_skills", []),
            "stipend":        intern.get("stipend", 0),
            "duration_weeks": intern.get("duration_weeks", 8),
            "capacity":       intern.get("capacity", 5),
            "score":          round(float(proba), 4),
            "score_pct":      round(float(proba) * 100, 1),
            "features":       fv_dict,
            "explanation":    explanation,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]




if __name__ == "__main__":
    from data import build_dataset, init_db

    # Data generation
    print("Step 1/3 — Generating synthetic dataset …")
    df = build_dataset(200)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/training_data.csv", index=False)
    print(f"  ✓ Saved data/training_data.csv  ({len(df)} rows)")

    # init DB
    print("\nStep 2/3 — Initialising database …")
    init_db()
    print("  ✓ internships.db ready")

    # Train
    print("\nStep 3/3 — Training model …")
    pipeline, meta = train(model_type="logistic")

    # smoke test
    print("\n🔍 Smoke test …")
    from data import get_internships
    interns = get_internships()[:10]
    test_user = {
        "education": "Bachelor's",
        "skills": ["Python", "Machine Learning", "Data Analysis"],
        "sector_interest": "tech",
        "location": "Bangalore",
        "category": "General",
        "district_type": "Urban",
        "prev_internship": 1,
    }
    recs = recommend(test_user, interns, top_n=3)
    for r in recs:
        print(f"  [{r['score_pct']:5.1f}%] {r['title']}  |  {r['location']}")
    print("\n✅ All done! Run: uvicorn main:app --reload")
