"""
Attention Visualizer Page

Shows attention heatmaps, temporal importance, and
gradient-based feature attribution from the trained LSTM model.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.model_registry import get_registry
from src.data.synthetic import FEATURE_NAMES, TARGET_NAMES


def show():
    st.title("Attention Visualizer")
    st.markdown("Explore what the LSTM model focuses on when making predictions.")

    registry = get_registry(st.session_state)

    if not registry.is_trained or "LSTM" not in registry.models:
        st.warning("Train models first on the **Model Comparison** page.")
        return

    lstm_result = registry.models["LSTM"]
    model = lstm_result.model
    X_val = registry.X_val
    y_val = registry.y_val

    # --- Sample Selector ---
    sample_idx = st.slider("Select validation sample", 0, len(X_val) - 1, 0)
    sample = torch.FloatTensor(X_val[sample_idx:sample_idx + 1])

    # Get attention weights
    model.eval()
    with torch.no_grad():
        output, attn_weights = model.forward_with_attention(sample)

    attn = attn_weights[0].numpy()  # (seq_len, seq_len)
    pred = output[0].numpy()
    actual = y_val[sample_idx]

    # --- Prediction Summary ---
    st.subheader("Prediction vs Actual")
    cols = st.columns(len(TARGET_NAMES))
    for i, (col, name) in enumerate(zip(cols, TARGET_NAMES)):
        with col:
            st.metric(name.replace("_", " ").title(),
                      f"{pred[i]:.3f}",
                      delta=f"{pred[i] - actual[i]:.3f}")

    # --- Attention Heatmap ---
    _show_attention_heatmap(attn)

    # --- Temporal Importance ---
    _show_temporal_importance(attn)

    # --- Gradient-based Feature Attribution ---
    _show_feature_attribution(model, sample)


def _show_attention_heatmap(attn: np.ndarray):
    """Display attention weight heatmap."""
    st.subheader("Attention Heatmap")
    st.caption("Shows how much each timestep attends to every other timestep.")

    seq_len = attn.shape[0]
    labels = [f"T-{seq_len - i}" for i in range(seq_len)]

    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=labels,
        y=labels,
        colorscale="Blues",
        colorbar=dict(title="Weight"),
    ))
    fig.update_layout(
        xaxis_title="Key (attended to)",
        yaxis_title="Query (attending from)",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_temporal_importance(attn: np.ndarray):
    """Bar chart of aggregated importance per timestep."""
    st.subheader("Temporal Importance")
    st.caption("Sum of attention received by each timestep (column sums of heatmap).")

    importance = attn.sum(axis=0)
    importance = importance / importance.sum()
    seq_len = len(importance)
    labels = [f"T-{seq_len - i}" for i in range(seq_len)]

    fig = go.Figure(data=go.Bar(
        x=labels, y=importance,
        marker_color="#636EFA",
    ))
    fig.update_layout(
        yaxis_title="Relative Importance",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_feature_attribution(model, sample: torch.Tensor):
    """Gradient-based feature importance via input gradients."""
    st.subheader("Feature Attribution (Gradient-based)")
    st.caption("Magnitude of input gradients shows which features the model is most sensitive to.")

    target_idx = st.selectbox("Target for attribution", list(range(len(TARGET_NAMES))),
                               format_func=lambda i: TARGET_NAMES[i],
                               key="attr_target")

    sample_grad = sample.clone().requires_grad_(True)
    model.zero_grad()
    output = model(sample_grad)
    output[0, target_idx].backward()

    grads = sample_grad.grad[0].numpy()  # (seq_len, n_features)
    # Average absolute gradient across timesteps
    feat_importance = np.abs(grads).mean(axis=0)
    feat_importance = feat_importance / feat_importance.sum()

    # Sort by importance
    sorted_idx = np.argsort(feat_importance)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_vals = feat_importance[sorted_idx]

    fig = go.Figure(data=go.Bar(
        x=sorted_vals, y=sorted_names,
        orientation="h",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        xaxis_title="Relative Importance",
        height=450,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature x Timestep heatmap
    st.subheader("Feature x Timestep Attribution Map")
    abs_grads = np.abs(grads)
    seq_len = abs_grads.shape[0]
    time_labels = [f"T-{seq_len - i}" for i in range(seq_len)]

    fig2 = go.Figure(data=go.Heatmap(
        z=abs_grads.T,
        x=time_labels,
        y=FEATURE_NAMES,
        colorscale="Reds",
        colorbar=dict(title="|Gradient|"),
    ))
    fig2.update_layout(
        xaxis_title="Timestep",
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)
