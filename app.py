import os
import io
import uuid
import traceback
import base64


from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from flask import session

import joblib

# -------------------------
# App setup
# -------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")



os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def save_model_blob(model, scaler, features):
    fid = uuid.uuid4().hex[:8]
    fname = f"model_{fid}.joblib"
    joblib.dump(
        {"model": model, "scaler": scaler, "features": features},
        os.path.join(app.config["MODEL_FOLDER"], fname)
    )
    return fname

def allowed_model_file(filename):
    return filename.lower().endswith((".joblib", ".pkl", ".sav"))

def safe_to_numeric_df(df, features):
    return df[features].apply(pd.to_numeric, errors="coerce")

# -------------------------
# Pages
# -------------------------
@app.route("/")
def index():
    summary = session.get("latest_run")
    return render_template(
        "index.html",
        active="home",
        summary=summary
    )


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
# Synthetic Data Generator
# -------------------------
@app.route("/api/generate_custom", methods=["POST"])
def generate_custom():
    try:
        payload = request.get_json(force=True)

        features = [f.strip() for f in payload.get("features", "").split(",") if f.strip()]
        class_params = payload.get("class_params", [])
        samples = int(payload.get("samples", 1000))
        seed = int(payload.get("seed", 42))

        if not features or not class_params:
            return jsonify({"status": "error", "message": "Invalid input."})

        np.random.seed(seed)
        rows = []
        per_class = max(1, samples // len(class_params))

        for cp in class_params:
            for _ in range(per_class):
                row = {f: np.random.normal(cp["mean"], cp["std"]) for f in features}
                row["label"] = cp["name"]
                rows.append(row)

        df = pd.DataFrame(rows)
        fname = f"data_{uuid.uuid4().hex[:8]}.csv"
        df.to_csv(os.path.join(app.config["UPLOAD_FOLDER"], fname), index=False)

        csv_url = url_for("download_upload", filename=fname)

        # ===============================
        # ✅ STATE PERSISTENCE (ADDED)
        # ===============================
        session["latest_run"] = {
            "action": "Dataset Generated",
            "rows": len(df),
            "features": features,
            "classes": [c["name"] for c in class_params],
            "csv_url": csv_url
        }

        return jsonify({
            "status": "ok",
            "csv_url": csv_url,
            "n": len(df)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# -------------------------
# EDA
# -------------------------
@app.route("/api/eda", methods=["POST"])
def api_eda():
    try:
        csv_url = request.get_json(force=True)["csv_url"]
        fname = csv_url.split("/")[-1]
        df = pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], fname))

        numeric = df.select_dtypes(include=[np.number])
        target = "label" if "label" in df.columns else None

        summary = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": numeric.columns.tolist(),
            "target": target
        }

        images = {}

        # -------------------------------------------------
        # Class Distribution
        # -------------------------------------------------
        if target:
            fig, ax = plt.subplots()
            df[target].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Class Distribution")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["class_dist"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Histograms
        # -------------------------------------------------
        if len(numeric.columns) > 0:
            n_cols = min(3, len(numeric.columns))
            n_rows = (len(numeric.columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for i, col in enumerate(numeric.columns):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f"Histogram of {col}")

            for i in range(len(numeric.columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["histograms"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Box Plots
        # -------------------------------------------------
        if len(numeric.columns) > 0:
            n_cols = min(3, len(numeric.columns))
            n_rows = (len(numeric.columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for i, col in enumerate(numeric.columns):
                if target:
                    sns.boxplot(x=target, y=col, data=df, ax=axes[i])
                    axes[i].set_title(f"{col} by {target}")
                else:
                    sns.boxplot(y=col, data=df, ax=axes[i])
                    axes[i].set_title(col)

            for i in range(len(numeric.columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["boxplots"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Violin Plots (NEW)
        # -------------------------------------------------
        if target and len(numeric.columns) > 0:
            n_cols = min(3, len(numeric.columns))
            n_rows = (len(numeric.columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for i, col in enumerate(numeric.columns):
                sns.violinplot(x=target, y=col, data=df, ax=axes[i])
                axes[i].set_title(f"Violin Plot of {col}")

            for i in range(len(numeric.columns), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["violinplots"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Pair Plot
        # -------------------------------------------------
        pair_features = numeric.columns[:3]
        if len(pair_features) > 1:
            pair_df = df[pair_features].copy()
            if target:
                pair_df[target] = df[target]
                g = sns.pairplot(pair_df, hue=target, diag_kind="kde", corner=True)
            else:
                g = sns.pairplot(pair_df, diag_kind="kde", corner=True)

            g.fig.suptitle("Pair Plot of Features", y=1.02)
            buf = io.BytesIO()
            g.savefig(buf, format="png")
            images["pairplot"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(g.fig)

        # -------------------------------------------------
        # Correlation Heatmap
        # -------------------------------------------------
        if len(numeric.columns) > 1:
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, ax=ax)
            ax.set_title("Correlation Heatmap")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["correlation_heatmap"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Feature Means per Class (NEW)
        # -------------------------------------------------
        if target and len(numeric.columns) > 0:
            means = df.groupby(target)[numeric.columns].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            means.plot(kind="bar", ax=ax)
            ax.set_title("Feature Means per Class")
            ax.set_ylabel("Mean Value")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["feature_means"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Feature Standard Deviation (NEW)
        # -------------------------------------------------
        if len(numeric.columns) > 0:
            stds = numeric.std()
            fig, ax = plt.subplots(figsize=(8, 4))
            stds.plot(kind="bar", ax=ax)
            ax.set_title("Feature Standard Deviation")
            ax.set_ylabel("Std Dev")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["feature_std"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # -------------------------------------------------
        # Scatter Plot (Top 2 Features) (NEW)
        # -------------------------------------------------
        if len(numeric.columns) >= 2:
            x_col, y_col = numeric.columns[:2]
            fig, ax = plt.subplots(figsize=(6, 5))
            if target:
                sns.scatterplot(x=x_col, y=y_col, hue=target, data=df, ax=ax)
            else:
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)

            ax.set_title(f"{x_col} vs {y_col}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            images["scatter"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        return jsonify({
            "status": "ok",
            "summary": summary,
            "images": images
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})
@app.route("/api/state")
def api_state():
    return jsonify(session.get("latest_run", {}))


# -------------------------
# Training & Simulation
# -------------------------
@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        p = request.get_json(force=True)

        fname = p["csv_url"].split("/")[-1]
        df = pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], fname))

        features = [f.strip() for f in p["features"].split(",")]
        target = p.get("target", "label")
        test_size = float(p.get("test_split", 30)) / 100

        X = safe_to_numeric_df(df, features)
        y = df[target]

        df_clean = pd.concat([X, y], axis=1).dropna()
        X, y = df_clean[features], df_clean[target]

        strat = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "GaussianNB": GaussianNB(),
            "MLP": MLPClassifier(max_iter=500)
        }

        metrics = {}
        best_acc = -1
        best_model = None
        best_name = ""

        for name, m in models.items():
            m.fit(X_train_s, y_train)
            preds = m.predict(X_test_s)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            metrics[name] = {"accuracy": acc, "f1": f1}

            if acc > best_acc:
                best_acc = acc
                best_model = m
                best_name = name

        model_file = save_model_blob(best_model, scaler, features)

        return jsonify({
            "status": "ok",
            "best_model": best_name,
            "best_accuracy": best_acc,
            "metrics": metrics,
            "model_url": url_for("download_model", filename=model_file)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# -------------------------
# Prediction
# -------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        p = request.get_json(force=True)
        fname = p["model_url"].split("/")[-1]

        mdata = joblib.load(os.path.join(app.config["MODEL_FOLDER"], fname))
        model = mdata["model"]
        scaler = mdata["scaler"]
        features = mdata["features"]

        X = np.array([[float(p["features_map"][f]) for f in features]])
        Xs = scaler.transform(X)

        result = {"status": "ok", "prediction": str(model.predict(Xs)[0])}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[0]
            result["probabilities"] = {
                str(c): float(p) for c, p in zip(model.classes_, probs)
            }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

# -------------------------
# Downloads
# -------------------------
@app.route("/uploads/<path:filename>")
def download_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/models/<path:filename>")
def download_model(filename):
    return send_from_directory(app.config["MODEL_FOLDER"], filename)
# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run()

