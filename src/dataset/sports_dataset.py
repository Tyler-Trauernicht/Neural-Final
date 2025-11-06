import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class SportsDataset(Dataset):
    """Sports Image Dataset for 10-class classification"""

    CLASSES = [
        'baseball', 'basketball', 'football', 'golf', 'hockey',
        'rugby', 'swimming', 'tennis', 'volleyball', 'weightlifting'
    ]

    def __init__(self, data_dir, split='train', image_size=32, augment=True):
        """
        Args:
            data_dir (str): Path to dataset directory
            split (str): 'train' or 'valid'
            image_size (int): Target image size (32 or 64)
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size

        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # Load file paths and labels
        self.samples = self._load_samples()

        # Define transforms
        self.transform = self._get_transforms(augment and split == 'train')

    def _load_samples(self):
        """Load all image paths and their corresponding labels"""
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)

        for class_name in self.CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))

        return samples

    def _get_transforms(self, augment=False):
        """Get image transforms for preprocessing"""
        transforms_list = []

        # Resize
        transforms_list.append(transforms.Resize((self.image_size, self.image_size)))

        # Augmentation for training
        if augment:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])

        # Convert to tensor and normalize
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        distribution = {cls: 0 for cls in self.CLASSES}
        for _, label in self.samples:
            class_name = self.CLASSES[label]
            distribution[class_name] += 1
        return distribution


def get_dataloaders(data_dir, batch_size=32, image_size=32, num_workers=2):
    """
    Create train and validation dataloaders

    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for training
        image_size (int): Target image size
        num_workers (int): Number of worker processes

    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    # Create datasets
    train_dataset = SportsDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=True
    )

    val_dataset = SportsDataset(
        data_dir=data_dir,
        split='valid',
        image_size=image_size,
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Number of classes: {len(SportsDataset.CLASSES)}")

    # Print class distribution
    train_dist = train_dataset.get_class_distribution()
    print("\nTraining set distribution:")
    for cls, count in train_dist.items():
        print(f"  {cls}: {count}")

    return train_loader, val_loader, len(SportsDataset.CLASSES)


if __name__ == "__main__":
    # Test the dataset
    data_dir = "data"
    train_loader, val_loader, num_classes = get_dataloaders(data_dir)

    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        if batch_idx == 0:
            break