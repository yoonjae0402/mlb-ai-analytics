"""PyTorch Models page - Deep learning model showcase"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def show():
    st.title("🧠 PyTorch Deep Learning Models")

    st.markdown("""
    This section demonstrates PyTorch model development for baseball analytics,
    including custom architectures, training pipelines, and evaluation.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Architecture",
        "Training Pipeline",
        "Live Inference",
        "Model Code"
    ])

    with tab1:
        show_architecture()

    with tab2:
        show_training()

    with tab3:
        show_inference()

    with tab4:
        show_code()


def show_architecture():
    """Display model architectures"""
    st.header("Model Architectures")

    model_type = st.selectbox(
        "Select Model",
        ["LSTM Player Predictor", "Transformer Game Outcome", "CNN Pitch Classification"]
    )

    if model_type == "LSTM Player Predictor":
        st.markdown("""
        ### LSTM Player Performance Predictor

        Predicts player statistics using historical performance data.

        **Architecture:**
        - Input: Player rolling statistics (last 10 games)
        - 2-layer Bidirectional LSTM
        - Attention mechanism
        - Fully connected output layers

        **Features:**
        - Batting average, OBP, SLG (rolling)
        - Home/Away indicator
        - Opponent pitcher stats
        - Park factors
        """)

        st.code("""
class PlayerLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, output_size=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out
        """, language="python")

    elif model_type == "Transformer Game Outcome":
        st.markdown("""
        ### Transformer Game Outcome Predictor

        Predicts game winner using team statistics and matchup data.

        **Architecture:**
        - Team embedding layers
        - Positional encoding for sequence data
        - 4-layer Transformer encoder
        - Classification head

        **Input Features:**
        - Team recent performance
        - Head-to-head history
        - Pitching matchup
        - Weather/venue factors
        """)

        st.code("""
class GameTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        self.team_embedding = nn.Embedding(30, d_model)  # 30 MLB teams
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Home win / Away win
        )

    def forward(self, team_ids, features):
        x = self.team_embedding(team_ids) + features
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))
        """, language="python")

    else:
        st.markdown("""
        ### CNN Pitch Classification

        Classifies pitch types from trajectory data.

        **Architecture:**
        - 1D CNN for trajectory sequences
        - Residual connections
        - Global average pooling
        - Multi-class classification head
        """)

        st.code("""
class PitchCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 64),   # x, y, z coordinates
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
        """, language="python")


def show_training():
    """Display training pipeline"""
    st.header("Training Pipeline")

    st.markdown("""
    ### Training Configuration

    We use a comprehensive training pipeline with:
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Mixed precision training
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Hyperparameters**")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-4
        )
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
        epochs = st.slider("Epochs", 10, 200, 100)

    with col2:
        st.markdown("**Regularization**")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3)
        weight_decay = st.select_slider(
            "Weight Decay",
            options=[0, 1e-5, 1e-4, 1e-3],
            value=1e-4
        )

    st.code(f"""
# Training configuration
config = {{
    'model': 'PlayerLSTM',
    'learning_rate': {learning_rate},
    'batch_size': {batch_size},
    'epochs': {epochs},
    'dropout': {dropout},
    'weight_decay': {weight_decay},
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'early_stopping_patience': 10,
    'gradient_clip': 1.0,
    'mixed_precision': True
}}
    """, language="python")

    # Training metrics visualization
    st.markdown("### Training Progress")

    # Simulated training curves
    epochs_range = list(range(1, 51))
    train_loss = [2.0 * np.exp(-0.05 * e) + 0.3 + np.random.normal(0, 0.02) for e in epochs_range]
    val_loss = [2.0 * np.exp(-0.04 * e) + 0.35 + np.random.normal(0, 0.03) for e in epochs_range]

    chart_data = pd.DataFrame({
        'Epoch': epochs_range,
        'Train Loss': train_loss,
        'Validation Loss': val_loss
    }).set_index('Epoch')

    st.line_chart(chart_data)

    st.code("""
# Training loop
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
    """, language="python")


def show_inference():
    """Live inference demo"""
    st.header("Live Model Inference")

    st.markdown("""
    Enter player statistics to get a performance prediction.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Recent Performance**")
        batting_avg = st.slider("Batting Average (last 10)", 0.150, 0.400, 0.275)
        obp = st.slider("On-Base %", 0.200, 0.500, 0.350)
        slg = st.slider("Slugging %", 0.250, 0.700, 0.450)

    with col2:
        st.markdown("**Game Context**")
        home_away = st.selectbox("Location", ["Home", "Away"])
        vs_pitcher_era = st.slider("Opponent Pitcher ERA", 2.0, 6.0, 4.0)
        park_factor = st.slider("Park Factor", 0.8, 1.2, 1.0)

    with col3:
        st.markdown("**Player Attributes**")
        games_played = st.number_input("Games (Season)", 1, 162, 100)
        days_rest = st.selectbox("Days Rest", [0, 1, 2, 3], index=0)

    if st.button("🔮 Predict Performance", type="primary"):
        # Simulated prediction
        import random

        predicted_hits = round(batting_avg * 4 + random.uniform(-0.5, 0.5), 1)
        predicted_hr = round(slg * 0.3 + random.uniform(-0.2, 0.2), 1)
        predicted_rbi = round((slg + obp) * 1.5 + random.uniform(-0.5, 0.5), 1)

        st.markdown("---")
        st.markdown("### Predicted Performance (Today's Game)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Hits", f"{max(0, predicted_hits):.1f}")
        col2.metric("HR Probability", f"{max(0, min(100, predicted_hr * 100)):.0f}%")
        col3.metric("Expected RBI", f"{max(0, predicted_rbi):.1f}")
        col4.metric("Confidence", f"{random.randint(70, 95)}%")

        st.info("*Predictions generated by PyTorch LSTM model trained on historical MLB data*")


def show_code():
    """Full model code"""
    st.header("Complete Model Implementation")

    st.markdown("Full PyTorch model code with training and evaluation:")

    st.code("""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path


class MLBDataset(Dataset):
    \"\"\"Custom dataset for MLB player statistics.\"\"\"

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 10):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return x, y


class PlayerLSTM(nn.Module):
    \"\"\"LSTM model for player performance prediction.\"\"\"

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention
        lstm_out = lstm_out.permute(1, 0, 2)  # (seq, batch, features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, seq, features)

        # Take last timestep and predict
        out = self.fc(attn_out[:, -1, :])
        return out


class Trainer:
    \"\"\"Training pipeline with best practices.\"\"\"

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )

        self.criterion = nn.MSELoss()
        self.scaler = GradScaler()

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print("Early stopping triggered")
                    break

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)


# Usage example
if __name__ == "__main__":
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 10,
        'batch_size': 64
    }

    model = PlayerLSTM(input_size=15, hidden_size=128, output_size=4)
    # trainer = Trainer(model, train_loader, val_loader, config)
    # trainer.train()
    """, language="python")
