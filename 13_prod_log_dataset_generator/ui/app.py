"""Streamlit dashboard for the production log dataset generator."""

from __future__ import annotations

import sys

import streamlit as st

sys.stdout.reconfigure(encoding="utf-8")

from src.dataset_generator import generate_dataset, export_dataset
from src.log_miner import LogMiner
from src.models import LogSource, MiningConfig

st.set_page_config(page_title="Eval Dataset Generator", page_icon=":factory:", layout="wide")

st.title("Automated Eval Dataset Generator from Production Logs")
st.markdown("Mine production logs, cluster with HDBSCAN, and generate diverse evaluation datasets.")

with st.sidebar:
    st.subheader("Configuration")
    source = st.selectbox("Log Source", [s.value for s in LogSource], index=0)
    time_range = st.slider("Lookback (hours)", 1, 720, 168)
    max_logs = st.number_input("Max Logs", 100, 50000, 10000, step=100)
    min_cluster_size = st.number_input("HDBSCAN min_cluster_size", 2, 100, 5)
    min_samples = st.number_input("HDBSCAN min_samples", 1, 50, 3)
    items_per_cluster = st.number_input("Items per cluster", 1, 20, 3)
    conn_url = st.text_input("Connection URL", "")

if st.button("Mine and Generate Dataset", type="primary"):
    config = MiningConfig(
        source=LogSource(source),
        connection_url=conn_url,
        time_range_hours=time_range,
        max_logs=max_logs,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        items_per_cluster=items_per_cluster,
    )

    with st.spinner("Mining production logs..."):
        miner = LogMiner(config)
        logs = miner.mine()
        st.success(f"Mined {len(logs)} production logs")

    with st.spinner("Clustering and generating dataset..."):
        dataset = generate_dataset(logs, config)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logs", len(logs))
    col2.metric("Clusters", dataset.total_clusters)
    col3.metric("Noise Points", dataset.noise_points)
    col4.metric("Dataset Items", len(dataset.items))

    st.subheader("Coverage Score")
    st.progress(dataset.coverage_score)
    st.text(f"{dataset.coverage_score:.1%} of non-noise logs represented")

    st.subheader("Dataset Items")
    diff_col1, diff_col2, diff_col3 = st.columns(3)
    easy = [i for i in dataset.items if i.difficulty == "easy"]
    medium = [i for i in dataset.items if i.difficulty == "medium"]
    hard = [i for i in dataset.items if i.difficulty == "hard"]
    diff_col1.metric("Easy", len(easy))
    diff_col2.metric("Medium", len(medium))
    diff_col3.metric("Hard", len(hard))

    st.dataframe([
        {
            "item_id": item.item_id[:8],
            "query": item.query[:80],
            "cluster": item.cluster_id,
            "difficulty": item.difficulty,
        }
        for item in dataset.items
    ])

    if st.button("Export Dataset"):
        export_dataset(dataset, "data/eval_dataset.json")
        st.success("Dataset exported to data/eval_dataset.json")
