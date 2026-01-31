import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the training and validation loop for the MLB Classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = "cpu",
        checkpoint_dir: str = "./models/checkpoints",
        max_grad_norm: float = 1.0,
        use_scheduler: bool = True
    ):
        """
        Initialize model trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Initial learning rate
            device: Device to train on ("cpu" or "cuda")
            checkpoint_dir: Directory to save model checkpoints
            max_grad_norm: Maximum gradient norm for clipping (0 to disable)
            use_scheduler: Whether to use learning rate scheduler
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Gradient clipping
        self.max_grad_norm = max_grad_norm

        # Learning rate scheduler (ReduceLROnPlateau)
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            )
        else:
            self.scheduler = None

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int = 10,
        patience: int = 5
    ) -> Dict[str, Any]:
        """
        Execute training loop with early stopping.
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                self.optimizer.step()

                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation Phase
            val_results = self.evaluate(val_loader)
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0.0
        }

    def save_checkpoint(self, filename: str):
        """Save model state."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model state."""
        path = self.checkpoint_dir / filename
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded checkpoint from {path}")
        else:
            logger.warning(f"Checkpoint {path} not found")
