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


class Decoder(nn.Module):
    def __init__(self, conv_dim=64):
        super(Decoder, self).__init__()
        self.deconv1 = deconv(conv_dim * 32, conv_dim * 16, 4)
        self.deconv1_1 = deconv(conv_dim * 16, conv_dim * 8, 3, stride=1)
        self.deconv1_2 = deconv(conv_dim * 8, conv_dim * 8, 3, stride=1)
        self.deconv2 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2_1 = deconv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.deconv2_2 = deconv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.deconv3 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3_1 = deconv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.deconv3_2 = deconv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.deconv4 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4_1 = deconv(conv_dim, conv_dim, 3, stride=1)
        self.deconv4_2 = deconv(conv_dim, conv_dim, 3, stride=1)
        self.deconv5 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.deconv1(x.detach()), 0.05)  # (?, 1024, 14, 14)
        out = F.leaky_relu(self.deconv1_1(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.deconv1_2(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.deconv2(out),   0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.deconv2_1(out), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.deconv2_2(out), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.deconv3(out),   0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.deconv3_1(out), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.deconv3_2(out), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.deconv4(out),   0.05)  # (?, 64, 112, 112)
        out = F.leaky_relu(self.deconv4_1(out), 0.05)  # (?, 64, 112, 112)
        out = F.leaky_relu(self.deconv4_2(out), 0.05)  # (?, 64, 112, 112)
        out = F.tanh(self.deconv5(out))                # (?, 3, 224, 224)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = models.resnet101(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        self.finetune(allow=False)

    def forward(self, x):
        out = self.encoder(x)                          # (?, 2048, 7, 7)
        return out

    def finetune(self, allow=False):
        for param in self.encoder.parameters():
            param.requires_grad = True if allow else False

class Encoder_scratch(nn.Module):
    def __init__(self, image_size=224, conv_dim=64):
        super(Encoder_scratch, self).__init__()
        self.tconv1 = conv(3, conv_dim * 2, 4, bn=False)
        self.tconv2 = conv(conv_dim * 2, conv_dim * 2 * 2, 4)
        self.tconv3 = conv(conv_dim * 2 * 2, conv_dim * 2 * 4, 4)
        self.tconv4 = conv(conv_dim * 2 * 4, conv_dim * 2 * 8, 4)
        self.tconv5 = conv(conv_dim * 2 * 8, conv_dim * 2 * 16, 4)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.tconv1(x), 0.05)  # (?, 128, 112, 112)
        out = F.leaky_relu(self.tconv2(out), 0.05)  # (?, 256, 56, 56)
        out = F.leaky_relu(self.tconv3(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.tconv4(out), 0.05)  # (?, 1024, 14, 14)
        out = F.leaky_relu(self.tconv5(out), 0.05)  # (?, 1024, 7 , 7)
        return out

class Decoder_scratch(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, image_size=224, conv_dim=64):
        super(Decoder_scratch, self).__init__()
        self.tdeconv1 = deconv(conv_dim * 32, conv_dim * 16, 4)
        self.tdeconv1_1 = deconv(conv_dim * 16, conv_dim * 8, 3, stride=1)
        self.tdeconv1_2 = deconv(conv_dim * 8, conv_dim * 8, 3, stride=1)
        self.tdeconv2 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.tdeconv2_1 = deconv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.tdeconv2_2 = deconv(conv_dim * 4, conv_dim * 4, 3, stride=1)
        self.tdeconv3 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.tdeconv3_1 = deconv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.tdeconv3_2 = deconv(conv_dim * 2, conv_dim * 2, 3, stride=1)
        self.tdeconv4 = deconv(conv_dim * 2, conv_dim, 4)
        self.tdeconv4_1 = deconv(conv_dim, conv_dim, 3, stride=1)
        self.tdeconv4_2 = deconv(conv_dim, conv_dim, 3, stride=1)
        self.tdeconv5 = deconv(conv_dim, 3, 4, bn=False)


    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.tdeconv1(x), 0.05)  # (?, 1024, 14, 14)
        out = F.leaky_relu(self.tdeconv1_1(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.tdeconv1_2(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.tdeconv2(out), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.tdeconv2_1(out), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.tdeconv2_2(out), 0.05)  # (?, 256, 28, 28)
        out = F.leaky_relu(self.tdeconv3(out), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.tdeconv3_1(out), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.tdeconv3_2(out), 0.05)  # (?, 128, 56, 56)
        out = F.leaky_relu(self.tdeconv4(out), 0.05)  # (?, 64, 112, 112)
        out = F.leaky_relu(self.tdeconv4_1(out), 0.05)  # (?, 64, 112, 112)
        out = F.leaky_relu(self.tdeconv4_2(out), 0.05)  # (?, 64, 112, 112)
        out = F.sigmoid(self.tdeconv5(out))  # (?, 3, 224, 224)
        return out

class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, image_size=224, conv_dim=128):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 1, int(image_size / 32), 3, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 128, 112, 112)
        relu_1= out
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 256, 56, 56)
        relu_2 = out
        out = F.leaky_relu(self.conv3(out), 0.05)    # (?, 512, 28, 28)
        relu_3 = out
        out = F.leaky_relu(self.conv4(out), 0.05)    # (?, 1024, 14, 14)
        out = F.leaky_relu(self.conv5(out), 0.05)    # (?, 1024, 7 , 7)
        out = F.sigmoid(self.fc(out).squeeze())
        return out, relu_1 ,relu_2, relu_3

class Discriminator_cls(nn.Module):
    def __init__(self, image_size=224, conv_dim=128):
        super(Discriminator_cls, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 2, int(image_size / 32), 3, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 128, 112, 112)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 256, 56, 56)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 512, 28, 28)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 1024, 14, 14)
        out = F.leaky_relu(self.conv5(out), 0.05)  # (?, 1024, 7 , 7)
        out = self.fc(out).squeeze()
        return out

#class Discriminator(nn.Module):
#    """Discriminator containing 4 convolutional layers."""

#    def __init__(self, image_size=224, conv_dim=128):
#        super(Discriminator, self).__init__()
#        self.conv1 = conv(3, conv_dim, 4, bn=False)
#        self.conv1_2 = conv(conv_dim, conv_dim, 3, stride=1)
#        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
#        self.conv2_2 = conv(conv_dim * 2, conv_dim * 2, 3, stride=1)
#        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
#        self.conv3_2 = conv(conv_dim * 4, conv_dim * 4, 3, stride=1)
#        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
#        self.conv4_2 = conv(conv_dim * 8, conv_dim * 8, 3, stride=1)
#        self.conv5 = conv(conv_dim * 8, conv_dim * 8, 4)
#        self.conv5_2 = conv(conv_dim * 8, conv_dim * 8, 3, stride=1)
#        self.fc = conv(conv_dim * 8, 2, int(image_size / 32), 1, 0, False)

#    def forward(self, x):  # If image_size is 64, output shape is as below.
#        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 128, 112, 112)
#        out = F.leaky_relu(self.conv1_2(out), 0.05)  # (?, 128, 112, 112)
#        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 256, 56, 56)
#        out = F.leaky_relu(self.conv2_2(out), 0.05)  # (?, 256, 56, 56)
#        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 512, 28, 28)
#        out = F.leaky_relu(self.conv3_2(out), 0.05)  # (?, 512, 28, 28)
#        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 1024, 14, 14)
#        out = F.leaky_relu(self.conv4_2(out), 0.05)  # (?, 1024, 14, 14)
#        out = F.leaky_relu(self.conv5(out), 0.05)  # (?, 1024, 7 , 7)
#        out = F.leaky_relu(self.conv5_2(out), 0.05)  # (?, 1024, 7 , 7)
#        out = self.fc(out).squeeze()
#        return out
