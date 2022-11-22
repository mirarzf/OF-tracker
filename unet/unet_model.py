""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention

        # Path down 
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Path up 
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        if attention: 
            # Path down 
            self.maxpool = nn.MaxPool2d(2)

            # Attention gate 
            self.att1 = Attention(in_channels=512)
            self.att2 = Attention(in_channels=256)
            self.att3 = Attention(in_channels=128)
            self.att4 = Attention(in_channels=64)


        self.outc = OutConv(64, n_classes)

    def forward(self, x, attmap):
        # RGB input 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.attention: 
            # Attention map input 
            attmap1 = self.maxpool(attmap)
            attmap2 = self.maxpool(attmap1)
            attmap3 = self.maxpool(attmap2)
            attmap4 = self.maxpool(attmap3)

            # Attention modules 
            attres1 = self.att2(attmap1, x1)
            attres2 = self.att2(attmap2, x2)
            attres3 = self.att2(attmap3, x3)
            attres4 = self.att1(attmap4, x4)

            # Path up 
            x = self.up1(x5, attres4)
            x = self.up2(x, attres3)
            x = self.up3(x, attres2)
            x = self.up4(x, attres1)
        
        else: # Usual path up of U-Net without attention
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        logits = self.outc(x)
        return logits