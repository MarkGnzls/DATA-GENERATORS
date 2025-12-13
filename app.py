# app.py — Full updated application with Plotly-ready model comparison visuals
import os
import io
import uuid
import json
import traceback
import base64

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # not used for the new visuals, but keep headless safe
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import joblib

# -------------------------
# App setup
# -------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB upload safeguard

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def save_model_blob(model_obj, scaler, features):
    file_id = uuid.uuid4().hex[:8]
    fname = f"model_{file_id}.joblib"
    path = os.path.join(app.config["MODEL_FOLDER"], fname)
    joblib.dump({"model": model_obj, "scaler": scaler, "features": features}, path)
    return fname

def allowed_model_file(filename):
    if not filename:
        return False
    fn = filename.lower()
    return fn.endswith(".joblib") or fn.endswith(".pkl") or fn.endswith(".sav")

def df_sample_preview(df, n=5):
    sample = df.sample(min(n, len(df))).to_dict(orient="records")
    return sample

def safe_to_numeric_df(df, features):
    # Convert selected features to numeric, coerce errors -> NaN
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X

# -------------------------
# Pages
# -------------------------
@app.route("/")
def index():
    return render_template("index.html", active="home")

@app.route("/dataset_builder")
def dataset_builder():
    return render_template("dataset_builder.html", active="dataset")

@app.route("/algorithm_guide")
def algorithm_guide():
    return render_template("algorithm_guide.html", active="algo")

@app.route("/model_training")
def model_training():
    return render_template("model_training.html", active="train")

# -------------------------
# Synthetic generator (simple multi-class gaussian per-class)
# -------------------------
@app.route("/api/generate_custom", methods=["POST"])
def generate_custom():
    """
    Accept JSON:
    {
      features: "f1,f2,f3",
      class_params: [{"name":"A","mean":100,"std":10}, ...],
      samples: 5000,
      seed: 42
    }
    Returns:
    {status:"ok", csv_url:"/uploads/...", n: N}
    """
    try:
        payload = request.get_json(force=True)
        features_raw = payload.get("features", "")
        class_params = payload.get("class_params", [])
        samples = int(payload.get("samples", 1000))
        seed = int(payload.get("seed", 42))

        features = [f.strip() for f in features_raw.split(",") if f.strip()]
        if not features:
            return jsonify({"status":"error","message":"No features provided."})

        if not class_params:
            return jsonify({"status":"error","message":"No class parameters provided."})

        np.random.seed(seed)
        records = []
        per_class = max(1, samples // len(class_params))

        for cp in class_params:
            name = cp.get("name")
            mean = float(cp.get("mean", 0))
            std = float(cp.get("std", 1))
            for _ in range(per_class):
                row = {f: np.random.normal(mean, std) for f in features}
                row["label"] = name
                records.append(row)

        df = pd.DataFrame(records)
        # Save
        fid = uuid.uuid4().hex[:8]
        fname = f"data_{fid}.csv"
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        df.to_csv(path, index=False)

        csv_url = url_for("download_upload", filename=fname)
        return jsonify({"status":"ok", "csv_url": csv_url, "n": len(df)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": str(e)})

# -------------------------
# EDA (basic numeric EDA) - returns small PNGs for quick preview plus JSON summaries
# -------------------------
@app.route("/api/eda", methods=["POST"])
def api_eda():
    """
    Expects: { csv_url: "/uploads/file.csv" }
    Returns: { status:"ok", images: { histogram: base64_png, ... }, summary: {...} }
    """
    try:
        payload = request.get_json(force=True)
        csv_url = payload.get("csv_url")
        if not csv_url:
            return jsonify({"status":"error","message":"csv_url required"})

        filename = csv_url.split("/")[-1]
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(fpath):
            return jsonify({"status":"error","message":"file not found on server"})

        df = pd.read_csv(fpath)

        # Try determine target
        target = None
        if "label" in df.columns:
            target = "label"
        else:
            obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if obj_cols:
                target = obj_cols[-1]

        # simple numeric summary
        numeric = df.select_dtypes(include=[np.number])
        summary = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "numeric_cols": numeric.columns.tolist(),
            "target": target
        }

        # create a few small base64 PNG previews (class distribution & histograms)
        imgs = {}

        # class distribution (if target exists)
        if target:
            try:
                fig, ax = plt.subplots(figsize=(4,3))
                df[target].value_counts().plot(kind="bar", ax=ax, color="#4e79a7")
                ax.set_title("Class distribution")
                buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
                imgs["class_count"] = base64.b64encode(buf.read()).decode("utf-8"); plt.close(fig)
            except Exception:
                pass

        # histograms (numeric)
        if not numeric.empty:
            try:
                fig = numeric.hist(figsize=(8,4))
                # pandas returns array of axes; save to buffer via plt
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                imgs["histograms"] = base64.b64encode(buf.read()).decode("utf-8")
                plt.close()
            except Exception:
                pass

        return jsonify({"status":"ok", "images": imgs, "summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": str(e)})

# -------------------------
# TRAIN endpoint — trains many models, returns metrics + Plotly-ready visualizations
# -------------------------
@app.route("/api/train", methods=["POST"])
def api_train():
    """
    POST JSON:
    {
      csv_url: "/uploads/xxx.csv",
      features: "f1,f2,f3",
      target: "label" (optional — will infer if missing),
      test_split: 30 (percent),
      kfold: 0
    }
    Response contains:
      - metrics (per model)
      - best_model name + model_url
      - classification_report (text)
      - confusion_matrix numeric matrix and classes
      - original_sample, scaled_sample
      - split_info
      - visualizations (plotly-ready JSON)
    """
    try:
        payload = request.get_json(force=True)
        csv_url = payload.get("csv_url")
        features_raw = payload.get("features", "")
        target = payload.get("target", "").strip()
        test_split_pct = float(payload.get("test_split", 30))
        kfold = int(payload.get("kfold", 0))

        if not csv_url:
            return jsonify({"status":"error","message":"csv_url required"})

        filename = csv_url.split("/")[-1]
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(path):
            return jsonify({"status":"error","message":"CSV not found on server"})

        df = pd.read_csv(path)

        # Infer target if not provided
        if not target:
            if "label" in df.columns:
                target = "label"
            else:
                obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                if obj_cols:
                    target = obj_cols[-1]
                else:
                    return jsonify({"status":"error","message":"Target not provided and could not be inferred."})

        if target not in df.columns:
            return jsonify({"status":"error","message":f"Target column '{target}' missing in CSV"})

        features = [f.strip() for f in features_raw.split(",") if f.strip()]
        if not features:
            return jsonify({"status":"error","message":"Features required"})

        # Prepare X, y
        X = safe_to_numeric_df(df, features)
        y = df[target]

        # Drop rows with NaN in features or target
        df_combined = pd.concat([X, y], axis=1).dropna()
        X = df_combined[features]
        y = df_combined[target]

        # Train-test split
        test_size = test_split_pct / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models to train (choice C — many models)
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "GaussianNB": GaussianNB(),
            "MLP": MLPClassifier(max_iter=500)
        }

        metrics = {}
        model_outputs = {}  # store predictions & probabilities for visualization
        best_acc = -1.0
        best_model = None
        best_name = None

        for name, clf in classifiers.items():
            try:
                clf.fit(X_train_scaled, y_train)
                preds = clf.predict(X_test_scaled)
                acc = float(accuracy_score(y_test, preds))
                f1s = float(f1_score(y_test, preds, average="weighted"))

                metrics[name] = {"accuracy": acc, "f1": f1s}

                # fetch probabilities if available
                proba = None
                try:
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X_test_scaled)
                    elif hasattr(clf, "decision_function"):
                        # decision_function can be used but may return shape (n_samples,) or (n_samples, n_classes)
                        dfun = clf.decision_function(X_test_scaled)
                        # try to convert to pseudo-proba via softmax (only if shape matches)
                        if dfun is not None:
                            # if binary returns shape (n_samples,), convert to two-column
                            if len(dfun.shape) == 1:
                                dfun = np.vstack([-dfun, dfun]).T
                            # softmax
                            ex = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                            proba = ex / ex.sum(axis=1, keepdims=True)
                except Exception:
                    proba = None

                model_outputs[name] = {
                    "pred": preds.tolist(),
                    "proba": proba.tolist() if proba is not None else None,
                    "classes": clf.classes_.tolist() if hasattr(clf, "classes_") else None
                }

                if acc > best_acc:
                    best_acc = acc
                    best_model = clf
                    best_name = name

            except Exception as e:
                metrics[name] = {"error": str(e)}
                model_outputs[name] = {"error": str(e)}

        # Save best model
        if best_model is None:
            return jsonify({"status":"error","message":"No model trained successfully."})

        model_fname = save_model_blob(best_model, scaler, features)
        model_url = url_for("download_model", filename=model_fname)

        # classification report and confusion matrix for best model
        try:
            best_preds = model_outputs[best_name]["pred"]
            class_rep_text = classification_report(y_test, best_preds, zero_division=0)
            cm = confusion_matrix(y_test, best_preds)
            classes_order = np.unique(y_test).tolist()
        except Exception:
            class_rep_text = ""
            cm = None
            classes_order = np.unique(y_test).tolist()

        # Build Plotly-ready visualization payloads (JSON-friendly numeric arrays)
        visualizations = {}

        # 1) Model comparison bar chart (accuracy & f1)
        try:
            labels = []
            acc_vals = []
            f1_vals = []
            for k in classifiers.keys():
                labels.append(k)
                m = metrics.get(k, {})
                acc_vals.append(float(m.get("accuracy", 0)) if isinstance(m, dict) else 0.0)
                f1_vals.append(float(m.get("f1", 0)) if isinstance(m, dict) else 0.0)
            visualizations["model_comparison"] = {
                "labels": labels,
                "accuracy": acc_vals,
                "f1": f1_vals
            }
        except Exception:
            visualizations["model_comparison"] = None

        # 2) Learning curve (for best model) — compute train_sizes & scores
        try:
            # learning_curve expects estimator with fit/predict, use cv=5 (fast)
            train_sizes, train_scores, val_scores = learning_curve(best_model, X_train_scaled, y_train, cv=5, n_jobs=1,
                                                                   train_sizes=np.linspace(0.1, 1.0, 5), scoring=None)
            train_mean = np.mean(train_scores, axis=1).tolist()
            train_std = np.std(train_scores, axis=1).tolist()
            val_mean = np.mean(val_scores, axis=1).tolist()
            val_std = np.std(val_scores, axis=1).tolist()
            train_sizes_list = train_sizes.tolist()

            visualizations["learning_curve"] = {
                "train_sizes": train_sizes_list,
                "train_mean": train_mean,
                "train_std": train_std,
                "val_mean": val_mean,
                "val_std": val_std,
                "model": best_name
            }
        except Exception:
            visualizations["learning_curve"] = None

        # 3) ROC Curve & AUC (one-vs-rest) for best model if probabilities available
        try:
            # binarize y_test to compute multiclass ROC
            unique_classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=unique_classes)
            proba = None
            # try get proba recorded earlier
            p = model_outputs.get(best_name, {}).get("proba")
            if p is not None:
                proba = np.array(p)
            else:
                # try compute on best_model
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(X_test_scaled)
                elif hasattr(best_model, "decision_function"):
                    dfun = best_model.decision_function(X_test_scaled)
                    if len(dfun.shape) == 1:
                        dfun = np.vstack([-dfun, dfun]).T
                    ex = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                    proba = ex / ex.sum(axis=1, keepdims=True)

            roc_data = {"classes": unique_classes.tolist(), "curves": {}}
            if proba is not None:
                for i, cls in enumerate(unique_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data["curves"][str(cls)] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": float(roc_auc)
                    }
                visualizations["roc_curve"] = roc_data
            else:
                visualizations["roc_curve"] = None
        except Exception:
            visualizations["roc_curve"] = None

        # 4) Precision-Recall curves for best model (if proba)
        try:
            if visualizations.get("roc_curve") is not None:
                pr_data = {"classes": visualizations["roc_curve"]["classes"], "curves": {}}
                proba_arr = proba
                for i, cls in enumerate(visualizations["roc_curve"]["classes"]):
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba_arr[:, i])
                    pr_data["curves"][str(cls)] = {
                        "precision": precision.tolist(),
                        "recall": recall.tolist()
                    }
                visualizations["pr_curve"] = pr_data
            else:
                visualizations["pr_curve"] = None
        except Exception:
            visualizations["pr_curve"] = None

        # 5) Feature importance (if available for best model)
        try:
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
                visualizations["feature_importance"] = {
                    "features": features,
                    "importances": importances.tolist()
                }
            else:
                visualizations["feature_importance"] = None
        except Exception:
            visualizations["feature_importance"] = None

        # Confusion matrix numeric & classes
        cm_payload = None
        if cm is not None:
            cm_payload = {
                "matrix": cm.tolist(),
                "classes": classes_order
            }

        # original sample / scaled sample
        original_sample = df_combined.sample(min(5, len(df_combined))).to_dict(orient="records")
        scaled_arr = scaler.transform(df_combined[features])
        scaled_df = pd.DataFrame(scaled_arr, columns=features)
        scaled_df[target] = df_combined[target].values
        scaled_sample = scaled_df.sample(min(5, len(scaled_df))).to_dict(orient="records")

        split_info = {
            "total": int(len(df_combined)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_pct": f"{int((1-test_size)*100)}%",
            "test_pct": f"{int(test_size*100)}%"
        }

        return jsonify({
            "status": "ok",
            "best_model": best_name,
            "best_accuracy": float(best_acc),
            "metrics": metrics,
            "classification_report": class_rep_text,
            "confusion_matrix": cm_payload,
            "original_sample": original_sample,
            "scaled_sample": scaled_sample,
            "split_info": split_info,
            "model_url": url_for("download_model", filename=model_fname),
            "visualizations": visualizations
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)})

# -------------------------
# Upload model endpoint
# -------------------------
@app.route("/api/upload_model", methods=["POST"])
def api_upload_model():
    try:
        file = request.files.get("model_file")
        features_raw = request.form.get("features", "")
        if not file:
            return jsonify({"status":"error", "message":"No file uploaded."})
        if not allowed_model_file(file.filename):
            return jsonify({"status":"error", "message":"Unsupported model file type."})

        file_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(file.filename)[1] or ".joblib"
        fname = f"upload_{file_id}{ext}"
        save_path = os.path.join(app.config["MODEL_FOLDER"], fname)
        file.save(save_path)

        meta_features = []
        try:
            mdata = joblib.load(save_path)
            if isinstance(mdata, dict) and "features" in mdata:
                meta_features = mdata["features"]
        except Exception:
            meta_features = [f.strip() for f in features_raw.split(",") if f.strip()]

        return jsonify({"status":"ok", "model_url": url_for("download_model", filename=fname), "meta": {"features": meta_features}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message":str(e)})

# -------------------------
# Prediction endpoint
# -------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        model_url = payload.get("model_url")
        features_map = payload.get("features_map", {})

        if not model_url:
            return jsonify({"status":"error","message":"model_url required."})

        fname = model_url.split("/")[-1]
        fpath = os.path.join(app.config["MODEL_FOLDER"], fname)
        if not os.path.exists(fpath):
            return jsonify({"status":"error","message":"model file not found."})

        mdata = joblib.load(fpath)
        # mdata expected dict with keys model, scaler, features
        if isinstance(mdata, dict):
            model = mdata.get("model")
            scaler = mdata.get("scaler")
            feature_order = mdata.get("features")
        else:
            return jsonify({"status":"error","message":"Uploaded model metadata missing."})

        if not feature_order:
            return jsonify({"status":"error","message":"Model metadata missing feature order."})

        # Build input row in feature order
        try:
            X_row = [float(features_map[f]) for f in feature_order]
        except Exception:
            return jsonify({"status":"error","message":"All feature values must be numeric and include all features."})

        X = np.array([X_row], dtype=float)
        if scaler is not None:
            Xs = scaler.transform(X)
        else:
            Xs = X

        pred = model.predict(Xs)[0]
        result = {"status":"ok", "prediction": str(pred)}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[0]
            result["probabilities"] = {str(c): float(p) for c, p in zip(model.classes_, probs)}

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)})

# -------------------------
# Download routes
# -------------------------
@app.route("/uploads/<path:filename>")
def download_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

@app.route("/models/<path:filename>")
def download_model(filename):
    return send_from_directory(app.config["MODEL_FOLDER"], filename, as_attachment=False)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
