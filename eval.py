import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Encoder, Decoder, ConvMaskGenerator
import torchvision.models as models
import torch.nn.functional as func
from logger import Logger
import os


class Eval(object):
    def __init__(self, testloader):
        self.batch_size = 20
        self.testloader = testloader
        self.model_path = os.path.join('home', 'david', 'NoiseGAN', 'data', 'good2-16', 'best-generator.pth')
        self.decoder = Decoder(conv_dim=64).cuda()
        self.test_CNN = models.resnet50(pretrained=True)
        self.encoder = Encoder().cuda()

    def eval(self):
        assert os.path.isfile(self.model_path), "file does not exist in path %s" % self.model_path
        self.decoder.load_state_dict(torch.load(self.model_path))
        print('load pretrained model from %s' % self.model_path)

        j = 0
        correct = 0
        total = 0
        correct_meanscore = 0
        for image, label in self.testloader:
            j += 1
            im_test = Variable(image, volatile=True).cuda()
            img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
            label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
            for index in range(self.batch_size):
                label_mask[index, label[index]] = 1
            outputs = self.test_CNN(img_test_resized)

            _, predicted = torch.max(outputs.data, 1)
            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / self.batch_size
            correct_meanscore += c
            total += label.size(0)
            correct += (predicted.cpu() == label).sum()
        correct_meanscore /= len(self.testloader)
        print('Model Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        print('Model Mean Accuracy: %.4f' % correct_meanscore.data[0])

        j = 0
        correct = 0
        total = 0
        correct_meanscore = 0
        for im, la in self.testloader:
            j += 1
            im_test = Variable(im, volatile=True).cuda()
            img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
            label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
            for index in range(self.batch_size):
                label_mask[index, la[index]] = 1
            mask_test = self.decoder(self.encoder(img_test_resized))
            reconst_images = img_test_resized + mask_test
            outputs = self.test_CNN(reconst_images)

            _, predicted = torch.max(outputs.data, 1)
            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / self.batch_size
            correct_meanscore += c
            total += la.size(0)
            correct += (predicted.cpu() == la).sum()
        correct_meanscore /= len(self.testloader)
        print('Model Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        print('Model Mean Accuracy: %.4f' % correct_meanscore.data[0])
