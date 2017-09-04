import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Encoder, Decoder
import torchvision.models as models
import torch.nn.functional as func

import os


class Eval(object):
    def __init__(self, testloader):
        self.batch_size = 10
        self.testloader = testloader
        self.model_path = os.path.join('/home', 'choikoal', 'NoiseGAN_imgnet_stloss', 'data', 'best-generator.pth')
        self.d_model_path = os.path.join('home', 'choikoal', 'NoiseGAN_imgnet_stloss', 'data', 'best-discriminator.pth')

        self.decoder = Decoder(conv_dim=64).cuda()
        self.test_CNN = models.resnet101(pretrained=True)
        self.test_CNN.fc = nn.Linear(self.test_CNN.fc.in_features, 10)
        self.pre_cnn_path = os.path.join('/home', 'choikoal', 'NoiseGAN_imgnet_stloss', 'data', 'best-pre_resnet101.pth')
        self.test_CNN.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained model from %s' % self.pre_cnn_path)
        self.encoder = Encoder().cuda()
        self.test_CNN.cuda()
        self.decoder.cuda()


    def eval(self):
        assert os.path.isfile(self.model_path), "file does not exist in path %s" % self.model_path
        self.decoder.load_state_dict(torch.load(self.model_path))
        print('load pretrained model from %s' % self.model_path)

        total = 0
        correct = 0
        correct_meanscore = 0
        correct_i = 0
        correct_meanscore_i = 0

        j = 0
        for im, la in self.testloader:
            j += 1
            im_test = Variable(im, volatile=True).cuda()
            img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
            label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
            for index in range(self.batch_size):
                label_mask[index, la[index]] = 1
            mask_test = self.decoder(self.encoder(img_test_resized))
            reconst_images = img_test_resized + mask_test  # - img_test_resized*mask_test
            outputs = self.test_CNN(reconst_images)


            _, predicted = torch.max(outputs.data, 1)


            total += la.size(0)

            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / self.batch_size
            correct_meanscore += c
            correct += (predicted.cpu() == la).sum()


            if j % 10 == 0:
                torchvision.utils.save_image(img_test_resized.data.cpu(),
                                             './data/test_images_%d.jpg' % (j))
                torchvision.utils.save_image(mask_test.data.cpu(),
                                             './data/test_noise_%d.jpg' % (j))
                torchvision.utils.save_image(reconst_images.data.cpu(),
                                             './data/test_reconst_images_%d.jpg' % (j))
        correct_meanscore /= 50
        correct_meanscore_i /= 50

        print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        print('Mean Accuracy: %.4f' % correct_meanscore.data[0])