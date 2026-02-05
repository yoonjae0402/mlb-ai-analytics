"""
Ensemble Lab Page

Combine LSTM + XGBoost with tunable weights or stacking.
Visualize ensemble improvement and weight sensitivity.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go

from src.models.model_registry import get_registry
from src.models.ensemble import EnsemblePredictor
from src.data.synthetic import TARGET_NAMES


def show():
    st.title("Ensemble Lab")
    st.markdown("Combine model predictions to see if ensembles outperform individual models.")

    registry = get_registry(st.session_state)

    if not registry.is_trained or len(registry.models) < 2:
        st.warning("Train both models first on the **Model Comparison** page.")
        return

    # Get predictions from both models
    lstm_preds, xgb_preds = _get_predictions(registry)
    y_val = registry.y_val

    # --- Strategy Selector ---
    strategy = st.radio("Ensemble Strategy", ["Weighted Average", "Stacking (Ridge Meta-Learner)"],
                        horizontal=True)

    if strategy == "Weighted Average":
        _weighted_average_section(lstm_preds, xgb_preds, y_val, registry)
    else:
        _stacking_section(lstm_preds, xgb_preds, y_val, registry)


def _get_predictions(registry) -> tuple[np.ndarray, np.ndarray]:
    """Get cached predictions from both models."""
    key = "_ensemble_preds"
    if key not in st.session_state:
        lstm_model = registry.models["LSTM"].model
        xgb_model = registry.models["XGBoost"].model
        X_val = registry.X_val

        lstm_model.eval()
        with torch.no_grad():
            lstm_preds = lstm_model(torch.FloatTensor(X_val)).numpy()
        xgb_preds = xgb_model.predict(X_val)
        st.session_state[key] = (lstm_preds, xgb_preds)

    return st.session_state[key]


def _compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _weighted_average_section(lstm_preds, xgb_preds, y_val, registry):
    """Interactive weighted average with sliders."""
    st.subheader("Weight Tuning")

    lstm_weight = st.slider("LSTM Weight", 0.0, 1.0, 0.5, 0.05)
    xgb_weight = 1.0 - lstm_weight

    st.caption(f"LSTM: {lstm_weight:.0%} | XGBoost: {xgb_weight:.0%}")

    ens = EnsemblePredictor(strategy="weighted_average", weights=[lstm_weight, xgb_weight])
    ens_preds = ens.predict([lstm_preds, xgb_preds])

    # --- Metrics Comparison ---
    _show_comparison_metrics(lstm_preds, xgb_preds, ens_preds, y_val)

    # --- Weight Sensitivity Curve ---
    _show_weight_sensitivity(lstm_preds, xgb_preds, y_val)


def _stacking_section(lstm_preds, xgb_preds, y_val, registry):
    """Stacking meta-learner section."""
    st.subheader("Stacking Meta-Learner")
    st.caption("A Ridge regression model learns optimal weights from a held-out set.")

    # Use training set predictions for fitting the meta-learner
    lstm_model = registry.models["LSTM"].model
    xgb_model = registry.models["XGBoost"].model
    X_train = registry.X_train
    y_train = registry.y_train

    lstm_model.eval()
    with torch.no_grad():
        lstm_train_preds = lstm_model(torch.FloatTensor(X_train)).numpy()
    xgb_train_preds = xgb_model.predict(X_train)

    ens = EnsemblePredictor(strategy="stacking")
    stacking_metrics = ens.fit_stacking([lstm_train_preds, xgb_train_preds], y_train)
    ens_preds = ens.predict([lstm_preds, xgb_preds])

    # --- Metrics Comparison ---
    _show_comparison_metrics(lstm_preds, xgb_preds, ens_preds, y_val)

    # Show stacking training info
    st.subheader("Stacking Meta-Learner Fit (Training Set)")
    cols = st.columns(len(TARGET_NAMES))
    for i, (col, name) in enumerate(zip(cols, TARGET_NAMES)):
        with col:
            st.metric(f"{name} MSE", f"{stacking_metrics[f'target_{i}_mse']:.4f}")


def _show_comparison_metrics(lstm_preds, xgb_preds, ens_preds, y_val):
    """Side-by-side metrics for all three approaches."""
    st.subheader("Metrics Comparison")

    lstm_mse = _compute_mse(y_val, lstm_preds)
    xgb_mse = _compute_mse(y_val, xgb_preds)
    ens_mse = _compute_mse(y_val, ens_preds)

    lstm_mae = _compute_mae(y_val, lstm_preds)
    xgb_mae = _compute_mae(y_val, xgb_preds)
    ens_mae = _compute_mae(y_val, ens_preds)

    best_individual_mse = min(lstm_mse, xgb_mse)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LSTM MSE", f"{lstm_mse:.4f}")
        st.metric("LSTM MAE", f"{lstm_mae:.4f}")
    with col2:
        st.metric("XGBoost MSE", f"{xgb_mse:.4f}")
        st.metric("XGBoost MAE", f"{xgb_mae:.4f}")
    with col3:
        improvement = ((best_individual_mse - ens_mse) / best_individual_mse * 100) if best_individual_mse > 0 else 0
        st.metric("Ensemble MSE", f"{ens_mse:.4f}",
                  delta=f"{improvement:.1f}% vs best" if improvement > 0 else f"{improvement:.1f}%")
        st.metric("Ensemble MAE", f"{ens_mae:.4f}")

    # Bar chart
    fig = go.Figure()
    models = ["LSTM", "XGBoost", "Ensemble"]
    mse_vals = [lstm_mse, xgb_mse, ens_mse]
    mae_vals = [lstm_mae, xgb_mae, ens_mae]

    fig.add_trace(go.Bar(x=models, y=mse_vals, name="MSE", marker_color="#636EFA"))
    fig.add_trace(go.Bar(x=models, y=mae_vals, name="MAE", marker_color="#EF553B"))

    fig.update_layout(barmode="group", height=400, template="plotly_white", yaxis_title="Error")
    st.plotly_chart(fig, use_container_width=True)


def _show_weight_sensitivity(lstm_preds, xgb_preds, y_val):
    """Plot ensemble MSE as a function of LSTM weight."""
    st.subheader("Weight Sensitivity")

    weights = np.linspace(0, 1, 51)
    mse_vals = []
    for w in weights:
        ens = EnsemblePredictor(strategy="weighted_average", weights=[w, 1 - w])
        preds = ens.predict([lstm_preds, xgb_preds])
        mse_vals.append(_compute_mse(y_val, preds))

    best_idx = int(np.argmin(mse_vals))
    best_w = weights[best_idx]
    best_mse = mse_vals[best_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weights, y=mse_vals,
        mode="lines", name="Ensemble MSE",
        line=dict(color="#636EFA", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[best_w], y=[best_mse],
        mode="markers+text", name="Optimal",
        marker=dict(color="red", size=12),
        text=[f"Best: {best_w:.0%} LSTM"],
        textposition="top center",
    ))

    fig.update_layout(
        xaxis_title="LSTM Weight",
        yaxis_title="MSE",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Optimal blend: **{best_w:.0%} LSTM / {1-best_w:.0%} XGBoost** (MSE: {best_mse:.4f})")
