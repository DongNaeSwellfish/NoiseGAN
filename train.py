import torch
import torch.nn as nn
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Generator
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as func
import torch.autograd as autograd
import numpy as np


class Trainer(object):
    def __init__(self, trainloader, testloader):
        self.num_gpu = 1
        self.batch_size = 40

        self.train_loader = trainloader
        self.test_loader = testloader

        self.generator = Generator(conv_dim=64).cuda()
        self.discriminator = Discriminator(image_size=96, conv_dim=64).cuda()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)
        self.finetune(allow=True)

        self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.001)
        self.optim_G = optim.Adam(self.generator.parameters(), lr=0.005)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.cnn.cuda()
            self.cnn.fc.cuda()

        self.criterion_C = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.MSELoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.real_label = 1
        self.fake_label = 0

    # if allow = True, classifier resnet50 computes grad
    def finetune(self, allow=True):
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.cnn.fc.parameters():
            param.requires_grad = True if allow else False

    # Train the fully-connected layer of resnet50 with STL10 dataset
    def train_classifier(self):
        total_step = len(self.train_loader)
        for epoch in range(5):
            for i, images in enumerate(self.train_loader):
                self.cnn.fc.zero_grad()
                images_label = Variable(images[1]).long().cuda()
                images = images[0].float().cuda()
                images = Variable(images)

                img_resized = func.upsample_bilinear(images, size=(224, 224))  # (96x96 -> 224x224)

                cnn_out = self.cnn(img_resized.detach())

                loss_fc = self.criterion_C(cnn_out, images_label)
                loss_fc.backward()  # make graph
                self.optim_C.step()  # update with gradient

                # evaluation with test dataset (800 per class)
                if (i % 124 == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    for im, la in self.test_loader:
                        # volatile means this Variable requires no grad computation
                        im_test = Variable(im, volatile=True).cuda()
                        test_img_resized = func.upsample_bilinear(im_test, size=(224, 224))

                        outputs = self.cnn(test_img_resized.detach())
                        _, predicted = torch.max(outputs.data, 1)

                        a = func.softmax(outputs)
                        b = torch.max(a, 1)
                        c = torch.mean(b[0])
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                    correct_meanscore /= 200  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

                print('Epoch [%d/5], Step[%d/%d], classification loss: %.4f, ' % (
                    epoch + 1, i + 1, total_step, loss_fc.data[0]))

    def train_adversarial(self):
        total_step = len(self.train_loader)
        for epoch in range(500):
            self.discriminator.train()
            self.generator.train()
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################
                i += 1
                labels_real = Variable(torch.zeros(self.batch_size).fill_(self.real_label)).long()
                labels_fake = Variable(torch.ones(self.batch_size).fill_(self.fake_label)).long()

                image_class = images[1].cuda()
                images = images[0].cuda()
                labels_real = labels_real.cuda()
                labels_fake = labels_fake.cuda()
                images = Variable(images)

                self.generator.zero_grad()
                self.discriminator.zero_grad()

                # Train discriminator with real image
                mask = self.generator(images)
                mask = mask * 0.01  # mask *= 0.01 is inplace operation(cannot compute gradient)
                logit_real = self.discriminator(images.detach())
                loss_real = self.criterion_D(logit_real, labels_real)
                loss_real.backward()

                # Train discriminator with fake image
                logit_fake = self.discriminator(images.detach()+mask)
                loss_fake = self.criterion_D(logit_fake, labels_fake)
                loss_fake.backward(retain_variables=True)

                self.optim_D.step()


                ######################################################
                #                  train Generator                   #
                ######################################################
                # Train generator with fake image with gradient uphill
                loss_G = -loss_fake
                loss_G.backward()
                self.optim_G.step()

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                self.cnn.fc.zero_grad()

                # Train generator with fake image with classifier (resnet50)
                mask = self.generator(images)
                mask = mask * 0.01

                img_resized = func.upsample_bilinear(images, size=(224, 224))
                mask_resized = func.upsample_bilinear(mask, size=(224, 224))

                cnn_out = self.cnn(img_resized.detach()+mask_resized)
                cnn_out = func.softmax(cnn_out)

                label_target = Variable(torch.ones((self.batch_size, 10))*0.1).cuda()
                # MSE loss with uniform distribution of 0.1
                loss_cls = self.criterion_G_CNN(cnn_out, label_target.detach())
                loss_cls.backward()
                self.optim_G.step()

                loss_g = loss_fake + loss_cls

                # Test the Model
                if (i % 124 == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    for im, la in self.test_loader:
                        im_test = Variable(im, volatile=True).cuda()
                        mask_test = self.generator(im_test)
                        mask_test = mask_test * 0.01

                        test_img_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        test_mask_resized = func.upsample_bilinear(mask_test, size=(224, 224))
                        outputs = self.cnn(test_img_resized + test_mask_resized)
                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = torch.max(a, 1)
                        c = torch.mean(b[0])
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                    correct_meanscore /= 200
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

                print('Epoch [%d/%d], Step[%d/%d], d_real_loss: %.4f, ''d_fake_loss: %.4f, g_loss: %.4f, cls_loss: %.4f'
                      % (epoch + 1, epoch, i + 1, total_step, loss_real.data[0], loss_fake.data[0], loss_g.data[0], loss_cls.data[0]))

