
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention.
    Learns the importance of feature channels.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        y = self.avg_pool(x)

        # Convert to (batch, 1, channels)
        y = y.squeeze(-1).transpose(-1, -2)

        # Learn channel attention weights
        y = self.conv(y)
        y = self.sigmoid(y)

        # Restore shape to (batch, channels, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Apply weights to the original feature maps
        return x * y.expand_as(x)


class OsteoarthritisModel(nn.Module):

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # Load EfficientNet-B0
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = efficientnet_b0(weights=weights)

        # EfficientNet-B0 produces 1280 feature channels
        self.eca = ECABlock(channels=1280)

        # Replace the original classifier
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.eca(features)

        # Global Average Pooling
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        # Five output scores: KL grades 0 through 4
        return self.classifier(features)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use pretrained ImageNet weights when available.
    # Set pretrained=False for an architecture-only test
    # that does not need to download weights.
    model = OsteoarthritisModel(
        num_classes=5,
        pretrained=False
    ).to(device)

    # Simulate a batch of 16 RGB images
    sample_images = torch.randn(16, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(sample_images)

    print("Device:", device)
    print("Input shape:", sample_images.shape)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])
    print("Model architecture test passed!")
