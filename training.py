import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from model import CIFAR10Netv2  # OLD CODE: missing import for model2
from dataset import get_data_loaders  # OLD CODE: missing import for dataset

class Trainer:
    """
    Flexible Trainer for CIFAR-10
    - Optimizers: Adam, SGD
    - Loss: CrossEntropy, NLLLoss
    - Scheduler: StepLR, OneCycleLR, ReduceLROnPlateau
    """

    def __init__(self, model, train_loader, val_loader, device=None,
                 optimizer_name="adam", loss_name="crossentropy", scheduler_name="reducelr",
                 lr=0.001, weight_decay=1e-4, momentum=0.9, step_size=10, gamma=0.1,
                 max_lr=0.01, patience=5):

        # OLD CODE: self.model = model.to(device)
        # NEW CODE: Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # -------------------------
        # Loss Function
        # -------------------------
        if loss_name.lower() == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name.lower() in ["nll", "nllloss", "negative_log_likelihood"]:
            self.criterion = nn.NLLLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        # -------------------------
        # Optimizer
        # -------------------------
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                       momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # -------------------------
        # Scheduler
        # -------------------------
        if scheduler_name.lower() == "steplr":
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == "onecyclelr":
            self.scheduler = OneCycleLR(self.optimizer, max_lr=max_lr,
                                        steps_per_epoch=len(train_loader), epochs=50)  # epochs can be param
        elif scheduler_name.lower() in ["reducelr", "reducelronplateau"]:
            # OLD CODE: patience=patience (default 5, too high - LR never reduces because accuracy keeps improving)
            # NEW CODE: patience=2 (reduce LR after 2 epochs without improvement)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=2)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        # -------------------------
        # Training history
        # -------------------------
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.learning_rates = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """Train one epoch with tqdm logging"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            acc = 100. * correct / total
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

        return running_loss / len(self.train_loader), 100. * correct / total

    def validate_epoch(self, epoch):
        """Validate one epoch with tqdm logging"""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                acc = 100. * correct / total
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

        return running_loss / len(self.val_loader), 100. * correct / total

    def train(self, epochs=50, target_acc=85.0):
        print(f"🚀 Training for {epochs} epochs on {self.device}")
        print("=" * 60)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)

            # Scheduler step (depends on type)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # Save history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # Save best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), "best_model.pth")
                tqdm.write(f"✅ New best model at epoch {epoch+1} ({val_acc:.2f}%)")

            # Epoch summary
            tqdm.write(
                f"📊 Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if val_acc >= target_acc:
                tqdm.write(f"\n🎉 Target accuracy {target_acc}% reached at epoch {epoch+1}")
                break

        print("=" * 60)
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        return self.best_val_acc


    def plot_history(self, save_path="training_curves.png"):
        """Plot loss, accuracy, and learning rate across epochs"""
        epochs = range(1, len(self.train_losses) + 1)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # ---- Loss curve ----
        axs[0].plot(epochs, self.train_losses, label="Train Loss")
        axs[0].plot(epochs, self.val_losses, label="Val Loss")
        axs[0].set_title("Loss per Epoch")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        # ---- Accuracy curve ----
        axs[1].plot(epochs, self.train_accuracies, label="Train Acc")
        axs[1].plot(epochs, self.val_accuracies, label="Val Acc")
        axs[1].axhline(y=85, color="r", linestyle="--", label="Target 85%")
        axs[1].set_title("Accuracy per Epoch")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].legend()
        axs[1].grid(True)

        # ---- Learning rate ----
        axs[2].plot(epochs, self.learning_rates, label="LR", color="g")
        axs[2].set_title("Learning Rate Schedule")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Learning Rate")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()      




def main():
    """Main training function"""
    # OLD CODE: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # NEW CODE: Set device with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # OLD CODE: model = create_model()
    # NEW CODE: Create model from model2.py
    model = CIFAR10Netv2(num_classes=10, dropout=0.1)
    
    # Print torch summary of the model
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    try:
        from torchsummary import summary
        # OLD CODE: summary(model, input_size=(3, 32, 32))
        # NEW CODE: Move model to device before summary for MPS compatibility
        model_for_summary = model.to(device)
        summary(model_for_summary, input_size=(3, 32, 32))
    except ImportError:
        print("torchsummary not available, printing model structure instead:")
        print(model)
    except Exception as e:
        print(f"Error with torchsummary (possibly MPS compatibility issue): {e}")
        print("Printing model structure instead:")
        print(model)
    print("="*60)
    
    # Get data loaders
    # OLD CODE: train_loader, val_loader = get_data_loaders(batch_size=128, num_workers=4)
    # NEW CODE: Adjust num_workers for MPS compatibility (MPS doesn't support multiprocessing well)
    num_workers = 0 if device.type == 'mps' else 4
    train_loader, val_loader = get_data_loaders(batch_size=128, num_workers=num_workers)
    
    # # Create trainer
    # trainer = Trainer(model, train_loader, val_loader, device)

    # OLD CODE: trainer = Trainer(model, train_loader, val_loader,
    #                  optimizer_name="adam",
    #                  loss_name="crossentropy",
    #                  scheduler_name="reducelr")
    # NEW CODE: Pass the detected device to the Trainer constructor
    trainer = Trainer(model, train_loader, val_loader, device=device,
                  optimizer_name="adam",
                  loss_name="crossentropy",
                  scheduler_name="reducelr")


    # OLD CODE: trainer.train(epochs=50, target_acc=85.0)
    # NEW CODE: Store the best accuracy returned by train method
    best_acc = trainer.train(epochs=50, target_acc=85.0)
    
    # # Train model
    # best_acc = trainer.train(epochs=100, target_acc=85.0)
    
    # OLD CODE: trainer.plot_training_history()
    # NEW CODE: Plot training history using correct method name
    trainer.plot_history()
    
    # Load best model and test
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # OLD CODE: val_loss, val_acc = trainer.validate_epoch()
    # NEW CODE: Final validation with proper epoch parameter
    val_loss, val_acc = trainer.validate_epoch(0)  # Use epoch 0 for final validation
    print(f"\nFinal validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")
    
    return best_acc


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training started at: {timestamp}")
    
    # Run training
    best_accuracy = main()
    
    print(f"\n{'='*50}")
    print(f"Training completed successfully!")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")
    print(f"{'='*50}")
