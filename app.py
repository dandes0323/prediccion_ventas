#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised ML App (Flask)
- Upload CSV
- Choose task: Clustering (KMeans) | Dimensionality Reduction (PCA 2D) | Anomaly Detection (IsolationForest)
- Auto-preprocess: numeric + categorical
- Visualize results
Run:
  pip install -r requirements.txt
  python app.py
Then open http://127.0.0.1:5000/
"""
import io
import os
import math
import uuid
from typing import List, Tuple

from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols)"""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocess_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    if not transformers:
        # Fallback to identity on all columns: convert to string and one-hot
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(df.columns)))  # noqa
    return ColumnTransformer(transformers=transformers, remainder="drop")

def figure_path(prefix="plot") -> str:
    return os.path.join(PLOT_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}.png")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        task = request.form.get("task", "kmeans")
        n_clusters = int(request.form.get("n_clusters", 3) or 3)
        contamination = float(request.form.get("contamination", 0.05) or 0.05)
        random_state = int(request.form.get("random_state", 42) or 42)

        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Sube un archivo CSV.", "error")
            return redirect(url_for("index"))

        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Error leyendo CSV: {e}", "error")
            return redirect(url_for("index"))

        if df.empty:
            flash("El CSV está vacío.", "error")
            return redirect(url_for("index"))

        # Detect columns
        numeric_cols, categorical_cols = detect_column_types(df)

        # Build preprocessing
        preprocess = build_preprocess_pipeline(numeric_cols, categorical_cols)

        # Fit-transform to feature matrix
        X = preprocess.fit_transform(df)

        result = {}
        plot_relpath = None

        if task == "kmeans":
            if X.shape[0] < n_clusters:
                flash(f"n_clusters ({n_clusters}) mayor que filas ({X.shape[0]}). Baja los clusters.", "error")
                return redirect(url_for("index"))
            model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            labels = model.fit_predict(X)
            df_out = df.copy()
            df_out["cluster"] = labels

            # PCA for visualization
            pca = PCA(n_components=2, random_state=random_state)
            X2 = pca.fit_transform(X)
            centers2 = pca.transform(model.cluster_centers_)

            # Plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(X2[:,0], X2[:,1], c=labels, alpha=0.7)
            ax.scatter(centers2[:,0], centers2[:,1], marker="X", s=120, edgecolor="k")
            ax.set_title("KMeans (PCA 2D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plot_path = figure_path("kmeans")
            fig.savefig(plot_path, bbox_inches="tight", dpi=130)
            plt.close(fig)
            plot_relpath = plot_path.replace(os.path.dirname(__file__) + os.sep, "")

            # Summaries
            result["task"] = "Clustering (KMeans)"
            result["n_samples"] = X.shape[0]
            result["n_features"] = X.shape[1]
            result["n_clusters"] = n_clusters
            # per-cluster counts
            counts = pd.Series(labels).value_counts().sort_index().to_dict()
            result["cluster_counts"] = counts

            # Prepare downloadable CSV
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
            csv_id = uuid.uuid4().hex[:10]
            fname = f"resultado_kmeans_{csv_id}.csv"
            path_csv = os.path.join(UPLOAD_DIR, fname)
            with open(path_csv, "wb") as f:
                f.write(csv_bytes.getvalue())
            result["download_url"] = url_for("download_file", filename=fname)

        elif task == "pca":
            # Project to 2D for visualization
            pca = PCA(n_components=2, random_state=random_state)
            X2 = pca.fit_transform(X)

            fig, ax = plt.subplots()
            ax.scatter(X2[:,0], X2[:,1], alpha=0.8)
            ax.set_title("PCA (2D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plot_path = figure_path("pca")
            fig.savefig(plot_path, bbox_inches="tight", dpi=130)
            plt.close(fig)
            plot_relpath = plot_path.replace(os.path.dirname(__file__) + os.sep, "")

            result["task"] = "Reducción de dimensionalidad (PCA)"
            result["explained_variance_ratio"] = [round(x, 4) for x in pca.explained_variance_ratio_.tolist()]
            result["n_samples"] = X.shape[0]
            result["n_features"] = X.shape[1]

            # Save transformed data
            df_out = df.copy()
            df_out["PC1"] = X2[:,0]
            df_out["PC2"] = X2[:,1]
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
            csv_id = uuid.uuid4().hex[:10]
            fname = f"resultado_pca_{csv_id}.csv"
            path_csv = os.path.join(UPLOAD_DIR, fname)
            with open(path_csv, "wb") as f:
                f.write(csv_bytes.getvalue())
            result["download_url"] = url_for("download_file", filename=fname)

        elif task == "isoforest":
            # Anomaly Detection
            model = IsolationForest(contamination=contamination, random_state=random_state)
            scores = model.fit_predict(X)  # 1 normal, -1 anomaly
            score_dec = model.decision_function(X)  # higher = more normal

            df_out = df.copy()
            df_out["anomaly"] = (scores == -1).astype(int)
            df_out["anomaly_score"] = score_dec

            # PCA for visualization
            pca = PCA(n_components=2, random_state=random_state)
            X2 = pca.fit_transform(X)

            fig, ax = plt.subplots()
            ax.scatter(X2[:,0], X2[:,1], c=df_out["anomaly"], alpha=0.8)
            ax.set_title("IsolationForest (PCA 2D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plot_path = figure_path("isof")
            fig.savefig(plot_path, bbox_inches="tight", dpi=130)
            plt.close(fig)
            plot_relpath = plot_path.replace(os.path.dirname(__file__) + os.sep, "")

            result["task"] = "Detección de anomalías (IsolationForest)"
            result["n_samples"] = X.shape[0]
            result["n_features"] = X.shape[1]
            result["contamination"] = contamination

            # Save results
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            csv_bytes = io.BytesIO(csv_buf.getvalue().encode("utf-8"))
            csv_id = uuid.uuid4().hex[:10]
            fname = f"resultado_isoforest_{csv_id}.csv"
            path_csv = os.path.join(UPLOAD_DIR, fname)
            with open(path_csv, "wb") as f:
                f.write(csv_bytes.getvalue())
            result["download_url"] = url_for("download_file", filename=fname)
        else:
            flash("Tarea no válida.", "error")
            return redirect(url_for("index"))

        return render_template("index.html", result=result, plot_relpath=plot_relpath)

    return render_template("index.html")

@app.route("/download/<path:filename>")
def download_file(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    # To allow running behind some free hosts that bind to PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
