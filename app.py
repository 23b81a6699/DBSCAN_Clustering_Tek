import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="DBSCAN Credit Card Clustering",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color:#0E1117;
}

.title {
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#00F5FF;
}

.subtitle {
    font-size:18px;
    text-align:center;
    color:white;
}

.stButton > button {
    background: linear-gradient(
    90deg,#ff416c,#ff4b2b);
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

dbscan = joblib.load("models/dbscan.pkl")
scaler = joblib.load("models/scaler.pkl")
labels = joblib.load("models/labels.pkl")

df = pd.read_csv("data/CreditCardDataset.csv")

# Remove Customer ID
if "CUST_ID" in df.columns:
    df.drop("CUST_ID", axis=1, inplace=True)

# Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.copy()

X_scaled = scaler.transform(X)

df["Cluster"] = labels

# --------------------------------------------------
# CLUSTER CENTERS
# --------------------------------------------------

cluster_centers = []

valid_clusters = sorted(
    [c for c in np.unique(labels) if c != -1]
)

for cluster in valid_clusters:

    cluster_centers.append(
        X_scaled[labels == cluster].mean(axis=0)
    )

cluster_centers = np.array(cluster_centers)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<p class='title'>💳 DBSCAN Credit Card Customer Segmentation</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Density-Based Customer Analysis & Outlier Detection</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("Customer Details")

balance = st.sidebar.number_input(
    "Balance",
    value=float(X["BALANCE"].median())
)

purchases = st.sidebar.number_input(
    "Purchases",
    value=float(X["PURCHASES"].median())
)

credit_limit = st.sidebar.number_input(
    "Credit Limit",
    value=float(X["CREDIT_LIMIT"].median())
)

payments = st.sidebar.number_input(
    "Payments",
    value=float(X["PAYMENTS"].median())
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.sidebar.button("Predict Cluster"):

    sample = pd.DataFrame({
        col: [X[col].mean()]
        for col in X.columns
    })

    sample["BALANCE"] = balance
    sample["PURCHASES"] = purchases
    sample["CREDIT_LIMIT"] = credit_limit
    sample["PAYMENTS"] = payments

    sample_scaled = scaler.transform(sample)

    distances = np.linalg.norm(
        cluster_centers - sample_scaled,
        axis=1
    )

    predicted_cluster = valid_clusters[
        np.argmin(distances)
    ]

    cluster_names = {
        0: "💳 Regular Customers",
        1: "💰 High Value Customers",
        2: "🛍 Frequent Buyers",
        3: "🏦 Premium Card Holders",
        4: "📈 Active Spenders",
        5: "⭐ Elite Customers"
    }

    st.markdown(
    f"""
    <div style="
    background:linear-gradient(
    90deg,#11998e,#38ef7d);
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-top:10px;
    ">

    <h3 style="color:white;">
    Predicted Cluster: {predicted_cluster}
    </h3>

    <h4 style="color:white;">
    {cluster_names.get(predicted_cluster,'Customer Segment')}
    </h4>

    </div>
    """,
    unsafe_allow_html=True
    )

# --------------------------------------------------
# METRICS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Customers",
        len(df)
    )

with c2:
    st.metric(
        "Features",
        X.shape[1]
    )

with c3:
    st.metric(
        "Clusters",
        len(valid_clusters)
    )

    with st.expander("❓ Reason"):
        st.write(
            """
            DBSCAN groups data based on density.

            Since most customers have similar spending and
            credit behavior, the algorithm detected one
            large dense group. Different values of eps and
            min_samples may create additional clusters or
            reveal more outliers.
            """
        )

with c4:
    st.metric(
        "Outliers",
        np.sum(labels == -1)
    )

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------

st.subheader("📊 Customer Segments (PCA Projection)")

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": pca_data[:,0],
    "PC2": pca_data[:,1],
    "Cluster": labels.astype(str)
})

fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    template="plotly_dark",
    title="DBSCAN Clusters & Noise Points"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# OUTLIER ANALYSIS
# --------------------------------------------------

st.subheader("🚨 Outlier Detection")

noise_count = np.sum(labels == -1)

st.info(
    f"Detected {noise_count} customers as outliers (Noise Points)"
)

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("📈 Cluster Distribution")

cluster_counts = pd.Series(labels).value_counts()

fig2 = px.bar(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values,
    labels={
        "x":"Cluster",
        "y":"Customer Count"
    },
    template="plotly_dark"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# --------------------------------------------------
# SILHOUETTE SCORE
# --------------------------------------------------

st.markdown("---")

if len(set(labels)) > 1:

    try:

        score = silhouette_score(
            X_scaled,
            labels
        )

        st.markdown(
        f"""
        <div style="
        background:linear-gradient(
        90deg,#6A11CB,#2575FC);
        padding:15px;
        border-radius:12px;
        text-align:center;
        ">
        <h2 style="color:white;">
        📊 Silhouette Score : {score:.4f}
        </h2>
        </div>
        """,
        unsafe_allow_html=True
        )

        with st.expander(
            "📘 What is Silhouette Score?"
        ):

            st.write(f"""
            Silhouette Score measures how well the clusters are separated.

            🟢 0.70 - 1.00 → Excellent

            🟡 0.50 - 0.70 → Good

            🟠 0.25 - 0.50 → Fair

            🔴 Below 0.25 → Poor

            Current Score: {score:.4f}
            """)

    except:
        st.warning(
            "Silhouette Score cannot be calculated."
        )