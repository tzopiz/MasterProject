#!/usr/bin/env python3
"""
TMJ Detector Model

3D CNN for regressing TMJ coordinates from full CBCT scans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TMJDetector(nn.Module):
    """
    3D CNN for TMJ coordinate regression
    
    Architecture:
        - 3D CNN encoder (progressively downsample)
        - Global average pooling
        - FC layers for coordinate regression
    
    Input: (B, 1, 96, 128, 128) - downsampled CBCT
    Output: (B, 6) - [left_z, left_y, left_x, right_z, right_y, right_x] normalized to [0,1]
    """
    
    def __init__(self, in_channels=1, features=[16, 32, 64, 128]):
        super(TMJDetector, self).__init__()
        
        # Encoder (3D convolutions with progressivedownsampling)
        self.encoder = nn.ModuleList()
        
        prev_channels = in_channels
        for feat in features:
            self.encoder.append(nn.Sequential(
                nn.Conv3d(prev_channels, feat, kernel_size=3, padding=1),
                nn.BatchNorm3d(feat),
                nn.ReLU(inplace=True),
                nn.Conv3d(feat, feat, kernel_size=3, padding=1),
                nn.BatchNorm3d(feat),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            ))
            prev_channels = feat
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(features[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        # Encoder
        for enc_block in self.encoder:
            x = enc_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Regression
        coords = self.fc(x)
        
        return coords


class TMJDetectorLarge(nn.Module):
    """
    Larger version with more parameters for better accuracy
    """
    
    def __init__(self, in_channels=1):
        super(TMJDetectorLarge, self).__init__()
        
        # Deeper encoder
        self.conv1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        
        self.conv5 = self._conv_block(256, 512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Regression head with skip connection
        self.fc_left = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
        
        self.fc_right = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.conv5(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Separate heads for left and right
        left_coords = self.fc_left(x)
        right_coords = self.fc_right(x)
        
        coords = torch.cat([left_coords, right_coords], dim=1)
        
        return coords


def get_detector_model(model_type='small', pretrained=None):
    """
    Factory function to get detector model
    
    Args:
        model_type: 'small' or 'large'
        pretrained: path to pretrained weights (optional)
    
    Returns:
        model: TMJDetector instance
    """
    if model_type == 'small':
        model = TMJDetector()
    elif model_type == 'large':
        model = TMJDetectorLarge()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    
    return model


if __name__ == '__main__':
    # Test model
    model = TMJDetector()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(2, 1, 96, 128, 128)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

