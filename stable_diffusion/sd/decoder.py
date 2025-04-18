import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (Batch_size, features, height, width)

        residue = x

        n, c, h, w = x.shape

        #(Batch_size, features, height, width) -> (Batch_size, features, height * width)
        x = x.transpose(n, c, h * w)

        #(Batch_size,features, height* width) -> (Batch_size, height*width, features)
        x = self.attention(x)

        #( Batch_size, height*width, features) -> (Batch_size, features, height*width)
        x = x.transpose(-1, -2)

        #(Batch_size, features, height*width) -> (batch_size, features, height, width)
        x = x.view((n, c, h, w))

        x += residue

        return x


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.Conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, height, width)

        residual = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x  = F.silu(x)

        x = self.Conv_2(x)

        return x + self.residual_layer(residual)
    

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(

            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            #(Batch_size, 512, height/ 8, weight/8) -> (Batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(Batch_size, 512, height/8, width/8) -> (Batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512),
            VAE_ResidualBlock(512),
            VAE_ResidualBlock(512),

            #(Batch_size, 512, height/4, width/4) -> (Batch_size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, height/2, width/2 ) -> (Batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.silu(),

            #(Batch_size, 128, height, width) -> (Batch_size, 3, height, width)
            nn.Conv2d(128, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, 4, height/8, width/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        #(Batch_size, 3, height, width)
        return x




