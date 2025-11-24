"""
3D U-Net for TMJ segmentation

This model segments the entire TMJ joint in 3D from cropped volumes.
Input: 3D crop around TMJ (e.g., 128x128x128)
Output: 3D binary mask of the joint
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation of TMJ.
    
    Architecture:
        - Encoder: 3 levels of 3D convolutions + pooling
        - Bottleneck: deepest level
        - Decoder: 3 levels of upsampling + skip connections
        - Output: sigmoid activation for binary segmentation
    """
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (1 for binary mask)
            init_features: Number of features in first layer
        """
        super(UNet3D, self).__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = UNet3D._block(features * 4, features * 8, name="bottleneck")
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")
        
        # Final output layer
        self.conv_final = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, D, H, W]
               e.g., [batch, 1, 128, 128, 128]
        
        Returns:
            Output tensor [B, 1, D, H, W] with sigmoid activation
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        return torch.sigmoid(self.conv_final(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        """
        Basic building block: Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
        )


class UNet2_5D(nn.Module):
    """
    2.5D U-Net: Hybrid approach that processes 2D slices with 3D context.
    
    Lighter alternative to full 3D U-Net, faster training, less memory.
    Processes volume slice-by-slice but uses adjacent slices for context.
    """
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        """
        Args:
            in_channels: 3 (uses 3 adjacent slices as "RGB" channels)
            out_channels: 1 (binary mask)
            init_features: Number of features in first layer
        """
        super(UNet2_5D, self).__init__()
        
        from .segmentation_model import UNet  # Reuse 2D U-Net
        
        self.unet_2d = UNet(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x):
        """
        Process each slice with its neighbors
        
        Args:
            x: Input [B, D, H, W] - single channel volume
        
        Returns:
            Output [B, D, H, W] - segmented volume
        """
        batch_size, depth, height, width = x.shape
        output = torch.zeros_like(x)
        
        for d in range(depth):
            # Get 3 adjacent slices
            slices = []
            for offset in [-1, 0, 1]:
                idx = max(0, min(depth - 1, d + offset))
                slices.append(x[:, idx, :, :].unsqueeze(1))
            
            # Stack as "RGB" [B, 3, H, W]
            input_slice = torch.cat(slices, dim=1)
            
            # Process through 2D U-Net
            output_slice = self.unet_2d(input_slice)
            
            # Store result
            output[:, d, :, :] = output_slice.squeeze(1)
        
        return output.unsqueeze(1)  # Add channel dimension


if __name__ == "__main__":
    # Test 3D U-Net
    print("Testing 3D U-Net...")
    
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)  # Small for testing
    
    # Create dummy input
    batch_size = 2
    depth, height, width = 64, 64, 64  # Small for testing
    
    x = torch.randn(batch_size, 1, depth, height, width)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.min() >= 0 and output.max() <= 1, "Output should be in [0, 1]"
    
    print("\n✅ 3D U-Net test passed!")
    
    # Test 2.5D U-Net
    print("\nTesting 2.5D U-Net...")
    model_25d = UNet2_5D(in_channels=3, out_channels=1, init_features=16)
    
    x_25d = torch.randn(batch_size, depth, height, width)
    
    with torch.no_grad():
        output_25d = model_25d(x_25d)
    
    print(f"2.5D Output shape: {output_25d.shape}")
    print("✅ 2.5D U-Net test passed!")

