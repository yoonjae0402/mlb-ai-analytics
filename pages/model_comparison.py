"""
Model Comparison Page

Train LSTM and XGBoost side-by-side with real training,
interactive Plotly charts, and metrics comparison.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.model_registry import (
    get_registry, generate_data, train_lstm, train_xgboost,
)
from src.data.synthetic import TARGET_NAMES


def show():
    st.title("Model Comparison")
    st.markdown("Train LSTM and XGBoost on synthetic player data, then compare performance.")

    registry = get_registry(st.session_state)

    # --- Training Controls ---
    with st.expander("Training Configuration", expanded=not registry.is_trained):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_players = st.slider("Players (dataset size)", 100, 500, 200, 50)
            epochs = st.slider("LSTM Epochs", 10, 100, 30, 5)
        with col2:
            lstm_lr = st.select_slider("LSTM Learning Rate", [0.01, 0.005, 0.001, 0.0005, 0.0001], value=0.001)
            lstm_hidden = st.select_slider("LSTM Hidden Size", [32, 64, 128, 256], value=64)
        with col3:
            xgb_estimators = st.slider("XGBoost Trees", 50, 500, 200, 50)
            xgb_depth = st.slider("XGBoost Max Depth", 3, 10, 6)

    train_btn = st.button(
        "Train Models" if not registry.is_trained else "Re-train Models",
        type="primary",
        use_container_width=True,
    )

    if train_btn:
        _run_training(registry, n_players, epochs, lstm_lr, lstm_hidden, xgb_estimators, xgb_depth)

    if not registry.is_trained:
        st.info("Click **Train Models** above to begin. Training takes ~10-30 seconds.")
        return

    # --- Results ---
    _show_metrics_summary(registry)
    _show_training_curves(registry)
    _show_prediction_scatter(registry)
    _show_per_target_comparison(registry)


def _run_training(registry, n_players, epochs, lstm_lr, lstm_hidden, xgb_estimators, xgb_depth):
    """Execute training with progress tracking."""
    progress = st.progress(0, text="Generating synthetic data...")
    X_train, y_train, X_val, y_val = generate_data(n_players=n_players)
    registry.X_train = X_train
    registry.y_train = y_train
    registry.X_val = X_val
    registry.y_val = y_val

    # Train LSTM
    status_text = st.empty()
    progress.progress(5, text="Training LSTM...")

    def lstm_progress(epoch, total, train_loss, val_loss):
        pct = int(5 + 55 * (epoch + 1) / total)
        progress.progress(pct, text=f"LSTM Epoch {epoch+1}/{total} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    lstm_result = train_lstm(
        X_train, y_train, X_val, y_val,
        epochs=epochs, lr=lstm_lr, hidden_size=lstm_hidden,
        progress_callback=lstm_progress,
    )
    registry.models["LSTM"] = lstm_result

    # Train XGBoost
    progress.progress(65, text="Training XGBoost...")

    xgb_result = train_xgboost(
        X_train, y_train, X_val, y_val,
        n_estimators=xgb_estimators, max_depth=xgb_depth,
    )
    registry.models["XGBoost"] = xgb_result
    registry.is_trained = True

    progress.progress(100, text="Training complete!")
    st.success(f"Both models trained. LSTM: {lstm_result.train_time:.1f}s | XGBoost: {xgb_result.train_time:.1f}s")


def _show_metrics_summary(registry):
    """Display metrics cards."""
    st.subheader("Performance Summary")

    lstm = registry.models["LSTM"]
    xgb = registry.models["XGBoost"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("LSTM MSE", f"{lstm.metrics['mse']:.4f}")
    with col2:
        st.metric("LSTM MAE", f"{lstm.metrics['mae']:.4f}")
    with col3:
        st.metric("LSTM R²", f"{lstm.metrics['r2']:.4f}")
    with col4:
        st.metric("XGB MSE", f"{xgb.metrics['mse']:.4f}",
                  delta=f"{xgb.metrics['mse'] - lstm.metrics['mse']:.4f}")
    with col5:
        st.metric("XGB MAE", f"{xgb.metrics['mae']:.4f}",
                  delta=f"{xgb.metrics['mae'] - lstm.metrics['mae']:.4f}")
    with col6:
        st.metric("XGB R²", f"{xgb.metrics['r2']:.4f}",
                  delta=f"{xgb.metrics['r2'] - lstm.metrics['r2']:.4f}")


def _show_training_curves(registry):
    """Plot training and validation loss curves."""
    st.subheader("Training Curves")

    lstm = registry.models["LSTM"]
    xgb = registry.models["XGBoost"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("LSTM", "XGBoost"))

    fig.add_trace(go.Scatter(y=lstm.train_losses, name="LSTM Train", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_trace(go.Scatter(y=lstm.val_losses, name="LSTM Val", line=dict(color="#EF553B", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(y=xgb.train_losses, name="XGB Train", line=dict(color="#00CC96")), row=1, col=2)
    fig.add_trace(go.Scatter(y=xgb.val_losses, name="XGB Val", line=dict(color="#AB63FA", dash="dash")), row=1, col=2)

    fig.update_layout(height=400, template="plotly_white")
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def _show_prediction_scatter(registry):
    """Scatter plot of predictions vs actuals."""
    st.subheader("Predictions vs Actuals")

    lstm_model = registry.models["LSTM"].model
    xgb_model = registry.models["XGBoost"].model
    X_val = registry.X_val
    y_val = registry.y_val

    import torch
    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(torch.FloatTensor(X_val)).numpy()
    xgb_preds = xgb_model.predict(X_val)

    target_idx = st.selectbox("Target variable", list(range(len(TARGET_NAMES))),
                               format_func=lambda i: TARGET_NAMES[i])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_val[:, target_idx], y=lstm_preds[:, target_idx],
        mode="markers", name="LSTM",
        marker=dict(color="#636EFA", opacity=0.6, size=8),
    ))
    fig.add_trace(go.Scatter(
        x=y_val[:, target_idx], y=xgb_preds[:, target_idx],
        mode="markers", name="XGBoost",
        marker=dict(color="#00CC96", opacity=0.6, size=8),
    ))

    # Perfect prediction line
    mn, mx = float(y_val[:, target_idx].min()), float(y_val[:, target_idx].max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect",
        line=dict(color="gray", dash="dash"),
    ))

    fig.update_layout(
        xaxis_title="Actual", yaxis_title="Predicted",
        height=450, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_per_target_comparison(registry):
    """Grouped bar chart comparing MSE per target."""
    st.subheader("Per-Target MSE Comparison")

    lstm_pt = registry.models["LSTM"].metrics["per_target"]
    xgb_pt = registry.models["XGBoost"].metrics["per_target"]

    fig = go.Figure()
    names = list(lstm_pt.keys())
    lstm_vals = [lstm_pt[n]["mse"] for n in names]
    xgb_vals = [xgb_pt[n]["mse"] for n in names]

    fig.add_trace(go.Bar(x=names, y=lstm_vals, name="LSTM", marker_color="#636EFA"))
    fig.add_trace(go.Bar(x=names, y=xgb_vals, name="XGBoost", marker_color="#00CC96"))

    fig.update_layout(
        barmode="group", height=400, template="plotly_white",
        yaxis_title="MSE",
    )
    st.plotly_chart(fig, use_container_width=True)
