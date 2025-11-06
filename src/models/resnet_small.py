import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic ResNet block for small ResNet"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetSmall(nn.Module):
    """Small ResNet architecture optimized for CPU training"""

    def __init__(self, num_classes=10, input_size=32):
        """
        Args:
            num_classes (int): Number of output classes
            input_size (int): Input image size (32 or 64)
        """
        super(ResNetSmall, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)   # 64 channels
        self.layer2 = self._make_layer(128, 2, stride=2)  # 128 channels, downsample
        self.layer3 = self._make_layer(256, 2, stride=2)  # 256 channels, downsample

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        """Create a layer with multiple basic blocks"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def get_feature_maps(self, x):
        """Extract feature maps from different layers for visualization"""
        features = {}

        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x

        # ResNet layers
        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        return features


def create_resnet_small(num_classes=10, input_size=32):
    """Factory function to create ResNetSmall model"""
    model = ResNetSmall(
        num_classes=num_classes,
        input_size=input_size
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_resnet_small(num_classes=10, input_size=32)

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

        # Test feature extraction
        features = model.get_feature_maps(dummy_input)
        print("\nFeature map shapes:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")

    # Test with 64x64 input
    model_64 = create_resnet_small(num_classes=10, input_size=64)
    dummy_input_64 = torch.randn(batch_size, 3, 64, 64)

    with torch.no_grad():
        output_64 = model_64(dummy_input_64)
        print(f"\n64x64 model parameters: {model_64.get_num_parameters():,}")
        print(f"64x64 output shape: {output_64.shape}")