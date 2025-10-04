import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class CIFAR10Dataset:
    """
    CIFAR-10 Dataset with Albumentations augmentation
    """
    
    def __init__(self, root='./data', train=True, download=True):
        self.root = root
        self.train = train
        self.download = download
        
        # CIFAR-10 mean and std for normalization
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        
        # Create transforms
        self.transform = self._get_transforms()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _get_transforms(self):
        """Get data augmentation transforms using Albumentations"""
        
        if self.train:
            # Training transforms with augmentation
            transform = A.Compose([
                # Horizontal flip
                A.HorizontalFlip(p=0.5),
                
                # Shift, scale, and rotate
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # 10% shift
                    scale_limit=0.1,      # 10% scale
                    rotate_limit=15,      # 15 degrees rotation
                    p=0.5
                ),
                
                # Coarse dropout
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,  # Fill with 0 (black)
                    p=0.3
                ),
                
                # Normalize
                A.Normalize(mean=self.mean, std=self.std),
                
                # Convert to tensor
                ToTensorV2()
            ])
        else:
            # Validation transforms (no augmentation)
            transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
            
        return transform
    
    def _load_dataset(self):
        """Load CIFAR-10 dataset"""
        # Load raw CIFAR-10 dataset
        dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=self.train,
            download=self.download
        )
        
        return dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get item with custom transform"""
        image, label = self.dataset[idx]
        
        # Convert PIL to numpy array
        image = np.array(image)
        
        # Apply albumentations transform
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, label


def get_data_loaders(batch_size=128, num_workers=4, root='./data'):
    """
    Get training and validation data loaders
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        root: Root directory for dataset
    
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    
    # Create datasets
    train_dataset = CIFAR10Dataset(root=root, train=True, download=True)
    val_dataset = CIFAR10Dataset(root=root, train=False, download=True)
    
    # OLD CODE: pin_memory=True
    # NEW CODE: Disable pin_memory for MPS compatibility
    import torch
    pin_memory = not (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_class_names():
    """Get CIFAR-10 class names"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


if __name__ == "__main__":
    # Test the dataset
    print("Testing CIFAR-10 dataset with Albumentations...")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size=4, num_workers=0)
    
    # Test training loader
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Get a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("Dataset test completed successfully!")
