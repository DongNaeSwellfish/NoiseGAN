import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class Unet_Generator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Unet_Generator, self).__init__()
        self.conv1 =   conv(3, conv_dim, 4, bn=False)
        self.conv1_1 = conv(conv_dim, conv_dim, 3, stride=1)
        self.conv2 =   conv(conv_dim, conv_dim * 2, 4)
        self.conv2_1 = conv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.conv3 =   conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv3_1 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.conv4 =   conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv4_1 = conv(conv_dim * 8, conv_dim * 8, 3, stride=1)
        self.conv5 =   conv(conv_dim * 8, conv_dim * 16, 4)
        self.conv5_1 = conv(conv_dim * 16, conv_dim * 16, 3, stride=1)
        self.deconv1 = deconv(conv_dim * 16, conv_dim * 8, 4)
        self.deconv1_1 = deconv(conv_dim * 8, conv_dim * 8, 3, stride=1)
        self.deconv2 = deconv(conv_dim * 16, conv_dim * 4, 4)
        self.deconv2_1 = deconv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.deconv3 = deconv(conv_dim * 8, conv_dim * 2, 4)
        self.deconv3_1 = deconv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.deconv4 = deconv(conv_dim * 4, conv_dim, 4)
        self.deconv4_1 = deconv(conv_dim, conv_dim, 3, stride=1)
        self.deconv5 = deconv(conv_dim * 2, 3, 4, bn=False)

    def forward(self, x, res1, res2, res3, res4):
        out = F.leaky_relu(self.conv1(x), 0.05)                            # (?, 64, 112, 112)
        out = F.leaky_relu(self.conv1_1(out), 0.05)                       # (?, 64, 112, 112)
        out = F.leaky_relu(self.conv2(out), 0.05)                         # (?, 128, 56, 56)
        out = F.leaky_relu(self.conv2_1(out), 0.05)                       # (?, 128, 56, 56)
        out = F.leaky_relu(self.conv3(out), 0.05)                         # (?, 256, 28, 28)
        out = F.leaky_relu(self.conv3_1(out), 0.05)                       # (?, 256, 28, 28)
        out = F.leaky_relu(self.conv4(out), 0.05)                         # (?, 512, 14, 14)
        out = F.leaky_relu(self.conv4_1(out), 0.05)                       # (?, 512, 14, 14)
        out = F.leaky_relu(self.conv5(out), 0.05)                         # (?, 1024, 7, 7)
        out = F.leaky_relu(self.conv5_1(out), 0.05)                        # (?, 1024, 7, 7)

        out = F.leaky_relu(self.deconv1(out), 0.05)                        # (?, 512, 14, 14)
        out = F.leaky_relu(self.deconv1_1(out), 0.05)                      # (?, 512, 14, 14)
        out = F.leaky_relu(self.deconv2(torch.cat((out, res4), 1)), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.deconv2_1(out), 0.05)                      # (?, 256, 28, 28)
        out = F.leaky_relu(self.deconv3(torch.cat((out, res3), 1)), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.deconv3_1(out), 0.05)                      # (?, 128, 56, 56)
        out = F.leaky_relu(self.deconv4(torch.cat((out, res2), 1)), 0.05)  # (?, 64, 112, 112)
        out = F.leaky_relu(self.deconv4_1(out), 0.05)                      # (?, 64, 112, 112)
        out = F.tanh(self.deconv5(torch.cat((out, res1), 1)))              # (?, 3, 224, 224)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.finetune(allow=False)

    def forward(self, x):
        out = self.encoder(x)                          # (?, 2048, 7, 7)
        return out

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False


class Discriminator_cls(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=224, conv_dim=64):
        super(Discriminator_cls, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, conv_dim * 16, 4)
        self.fc = conv(conv_dim * 16, 2, int(image_size / 32), 1, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        res1 = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 112, 112)
        res2 = F.leaky_relu(self.conv2(res1), 0.05)    # (?, 128, 56, 56)
        res3 = F.leaky_relu(self.conv3(res2), 0.05)    # (?, 256, 28, 28)
        res4 = F.leaky_relu(self.conv4(res3), 0.05)    # (?, 512, 14, 14)
        res5 = F.leaky_relu(self.conv5(res4), 0.05)    # (?, 1024, 7 , 7)
        out = self.fc(res5).squeeze()
        return out, res1, res2, res3, res4