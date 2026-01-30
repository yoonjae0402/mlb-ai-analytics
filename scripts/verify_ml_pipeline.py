import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PlayerPerformanceLSTM, ModelTrainer, PredictionExplainer

def verify_pipeline():
    print("⚾️ Verifying MLB Video Pipeline - ML Module ⚾️")
    
    # 1. Setup Data (Dummy)
    print("\n1. Generating dummy data...")
    batch_size = 8
    seq_len = 10
    features = 11
    classes = 3
    
    # Simulate 100 samples
    x = torch.randn(100, seq_len, features)
    y = torch.randint(0, classes, (100,))
    
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    
    # 2. Initialize Model
    print("\n2. Initializing LSTM Model...")
    model = PlayerPerformanceLSTM(input_size=features, num_classes=classes)
    print(f"   Model created: {model}")
    
    # 3. Train Model
    print("\n3. Starting Training Loop...")
    trainer = ModelTrainer(model, learning_rate=0.01)
    history = trainer.train(train_loader, val_loader, epochs=2)
    print("   Training complete.")
    print(f"   Final Loss: {history['train_loss'][-1]:.4f}")
    
    # 4. Explain a Prediction
    print("\n4. Testing Explainer...")
    explainer = PredictionExplainer()
    dummy_seq = np.random.rand(seq_len, features)
    # Force some high values for "Above Average" explanation
    dummy_seq[:, 0] = 0.350 # High BA
    
    explanation = explainer.explain_prediction(dummy_seq, prediction_class=2, confidence=0.92)
    print("   Explanation generated:")
    print(f"   Prediction: {explanation['prediction']}")
    print(f"   Reasons: {explanation['reasons']}")
    
    print("\n✅ ML Pipeline Verification Successful!")

if __name__ == "__main__":
    verify_pipeline()
