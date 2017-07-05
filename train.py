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
        self.batch_size = 25

        self.train_loader = trainloader
        self.test_loader = testloader

        self.generator = Generator(conv_dim=64).cuda()
        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)
        self.finetune(allow=True)

        self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.001)
        self.optim_G_dis = optim.Adam(self.generator.parameters(), lr=0.0005)
        self.optim_G_cls = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
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
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(1):
            for i, images in enumerate(self.train_loader):
                self.cnn.fc.zero_grad()
                images_label = Variable(images[1]).long().cuda()
                images = images[0].float().cuda()
                images = Variable(images)

                img_resized = func.upsample_bilinear(images, size=(224, 224))  # (96x96 -> 224x224)

                cnn_out = self.cnn(img_resized.detach())

                loss_fc = self.criterion_C(cnn_out, images_label)
                loss_fc.backward()  # make graph
                clip_gradient(self.optim_C, 0.5)
                self.optim_C.step()  # update with gradient

                if (i % 10) == 0:
                    print('Epoch [%d/5], Step[%d/%d], classification loss: %.4f, ' % (
                        epoch + 1, i, total_step, loss_fc.data[0]))

                # evaluation with test dataset (800 per class)
                if (i % 199 == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    for im, la in self.test_loader:
                        # volatile means this Variable requires no grad computation
                        im_test = Variable(im, volatile=True).cuda()
                        label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
                        for index in range(self.batch_size):
                            label_mask[index, la[index]] = 1
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        outputs = self.cnn(img_test_resized.detach())
                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / self.batch_size
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                    correct_meanscore /= 320  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

    def train_adversarial(self):
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(200):
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
                images_resized = func.upsample_bilinear(images, (224, 224))

                self.generator.zero_grad()
                self.discriminator.zero_grad()

                # Train discriminator with real image
                mask = func.tanh(self.generator(images_resized))
                mask = mask * 0.1  # mask *= 0.01 is inplace operation(cannot compute gradient)
                logit_real = self.discriminator(images_resized.detach())
                loss_real_real = self.criterion_D(logit_real, labels_real)
                loss_real_real.backward()

                # Train discriminator with fake image
                logit_fake = self.discriminator(images_resized.detach()+mask)
                loss_fake_fake = self.criterion_D(logit_fake, labels_fake)
                loss_fake_fake.backward(retain_variables=True)

                if (i % 5) == 0:
                    clip_gradient(self.optim_D, 0.5)
                    self.optim_D.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                self.cnn.fc.zero_grad()
                logit_fake = self.discriminator(images_resized.detach() + mask)
                loss_fake_real = self.criterion_D(logit_fake, labels_real)
                loss_fake_real.backward()  #(retain_variables=True)
                clip_gradient(self.optim_G_dis, 0.5)
                self.optim_G_dis.step()

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                self.cnn.fc.zero_grad()

                # Train generator with fake image with classifier (resnet50)
                mask = func.tanh(self.generator(images_resized))
                mask = mask * 0.1
                cnn_out = self.cnn(images_resized.detach()+mask)
                cnn_out = func.softmax(cnn_out)

                label_target = Variable(torch.zeros((self.batch_size, 1))).cuda()
                # MSE loss with uniform distribution of 0.1
                one_mask = Variable(torch.zeros(self.batch_size, 10)).cuda()
                for index in range(self.batch_size):
                    one_mask[index, image_class[index]] = 1
                cnn_out_truth = one_mask.detach() * cnn_out
                cnn_out_truth = torch.sum(cnn_out_truth, dim=1)
                loss_cls = self.criterion_G_CNN(cnn_out_truth, label_target.detach())
                loss_cls.backward()
                clip_gradient(self.optim_G_cls, 0.5)
                self.optim_G_cls.step()

                if (i % 10) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_fake_real: %.4f, ''loss_real_real: %.4f, loss_fake_fake: %.4f, cls_loss: %.4f'
                          % (epoch + 1, epoch, i + 1, total_step, loss_fake_real.data[0], loss_real_real.data[0], loss_fake_fake.data[0], loss_cls.data[0]))

                # Test the Model
                if (i % 199 == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
                        for index in range(self.batch_size):
                            label_mask[index, la[index]] = 1
                        mask_test = func.tanh(self.generator(img_test_resized) * 0.1)
                        reconst_images = img_test_resized + mask_test
                        outputs = self.cnn(reconst_images)

                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / self.batch_size
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                        if j % 320 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(), './data/images_%d.jpg' % (epoch + 1))
                            torchvision.utils.save_image(mask_test.data.cpu(), './data/noise_%d.jpg' % (epoch + 1))
                            torchvision.utils.save_image(reconst_images.data.cpu(), './data/reconst_images_%d.jpg' % (epoch + 1))
                    correct_meanscore /= 320
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
