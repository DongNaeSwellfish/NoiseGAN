import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        self.conv1 = conv(3, conv_dim, 3, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 3)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 3)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 3)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 48, 48)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 24, 24)
        out = F.leaky_relu(self.conv3(out), 0.05)    # (?, 256, 12, 12)
        out = F.leaky_relu(self.conv4(out), 0.05)    # (?, 512, 6, 6)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 12, 12)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 24, 24)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 48, 48)
        out = F.tanh(self.deconv4(out))              # (?, 3, 96, 96)
        return out


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=96, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 2, int(image_size / 16), 1, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 48, 48)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 24, 24)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 12, 12)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 6, 6)
        out = self.fc(out).squeeze()
        return out
