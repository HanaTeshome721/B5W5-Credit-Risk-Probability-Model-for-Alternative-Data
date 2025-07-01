import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Calculate RFM Metrics
def calculate_rfm(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    snapshot = pd.to_datetime(snapshot_date)

    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm

# Step 2: Scale RFM
def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    return pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"])

# Step 3: Cluster Customers
def cluster_rfm(rfm_scaled: pd.DataFrame, original_rfm: pd.DataFrame, n_clusters=3, random_state=42) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    original_rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    return original_rfm

# Step 4: Analyze & Label High-Risk Cluster
def label_high_risk(rfm_with_clusters: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = rfm_with_clusters.groupby("Cluster").mean(numeric_only=True)
    high_risk_cluster = cluster_summary.sort_values(["Frequency", "Monetary"]).index[0]
    
    rfm_with_clusters["is_high_risk"] = (rfm_with_clusters["Cluster"] == high_risk_cluster).astype(int)
    return rfm_with_clusters[["CustomerId", "is_high_risk"]]

# Step 5: Full Proxy Pipeline
def generate_proxy_labels(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    rfm = calculate_rfm(df, snapshot_date)
    rfm_scaled = scale_rfm(rfm)
    rfm_with_clusters = cluster_rfm(rfm_scaled, rfm)
    labels = label_high_risk(rfm_with_clusters)
    return labels
