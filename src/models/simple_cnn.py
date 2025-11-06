import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN architecture for sports image classification"""

    def __init__(self, num_classes=10, input_size=32, dropout_rate=0.5):
        """
        Args:
            num_classes (int): Number of output classes
            input_size (int): Input image size (32 or 64)
            dropout_rate (float): Dropout probability
        """
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16 or 64x64 -> 32x32

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8 or 32x32 -> 16x16

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to 4x4
        )

        # Calculate the size after feature extraction
        self.feature_size = 128 * 4 * 4  # 128 channels * 4x4 spatial

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """Forward pass"""
        # Feature extraction
        x = self.features(x)

        # Flatten for classifier
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_num_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_cnn(num_classes=10, input_size=32, dropout_rate=0.5):
    """Factory function to create SimpleCNN model"""
    model = SimpleCNN(
        num_classes=num_classes,
        input_size=input_size,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_simple_cnn(num_classes=10, input_size=32)

    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test with 64x64 input
    model_64 = create_simple_cnn(num_classes=10, input_size=64)
    dummy_input_64 = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        output_64 = model_64(dummy_input_64)
        print(f"\n64x64 model parameters: {model_64.get_num_parameters():,}")
        print(f"64x64 output shape: {output_64.shape}")