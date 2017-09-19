import torch
import torch.nn as nn
import torchvision
#import numpy as np
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Encoder, Decoder
import torchvision.models as models
import torch.nn.functional as func
import os
from logger import Logger

logger = Logger('./logs')


class Trainer(object):
    def __init__(self, trainloader, testloader, opt):
        self.opt=opt
        self.num_gpu = 1
        self.batch_size = 20

        self.train_loader = trainloader
        self.test_loader = testloader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(conv_dim=64).cuda()

        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()

        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)

        #/home/yjyoo/Code/NoiseGAN-koal_stloss/data
        self.pre_cnn_path = os.path.join('/home', 'david', 'NoiseGAN', 'data', 'best-pre_resnet101.pth')
        self.cnn.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained model from %s' % self.pre_cnn_path)

        self.cnn_2 = models.resnet50(pretrained=True)
        self.cnn_2.fc = nn.Linear(self.cnn_2.fc.in_features, 10)

        self.pre_cnn_2_path = os.path.join('/home', 'david', 'NoiseGAN', 'data', 'best-pre_resnet50.pth')
        self.cnn_2.load_state_dict(torch.load(self.pre_cnn_2_path))
        print('load pretrained model from %s' % self.pre_cnn_2_path)

        #define imitator
        self.imitator = models.resnet34(pretrained=True)
        self.imitator.fc = nn.Linear(self.imitator.fc.in_features, 10)

        self.finetune(allow=True)

        self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        self.optim_G_dis = optim.Adam(self.decoder.parameters(), lr=0.0005)
        self.optim_G_cls = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_G = optim.Adam(self.decoder.parameters(), lr = 0.001)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.optim_L1 = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_I = optim.Adam(self.imitator.fc.parameters(), lr = 0.001)

        self.criterion_C = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()
        self.criterion_I = nn.SmoothL1Loss()

        self.real_label = 1
        self.fake_label = 0

        self.cls = 1

        self.l1_param = 0.001

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.cnn.cuda()
            self.cnn.fc.cuda()
            self.imitator.cuda()
            self.imitator.fc.cuda()
            self.cnn_2.cuda()
            self.cnn_2.fc.cuda()



    # if allow = True, classifier resnet50 computes grad
    def finetune(self, allow=True):
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

    # Train the fully-connected layer of resnet50 with STL10 dataset
    def train_classifier(self, opt):
        best_score = 0

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(2):
            for i, images in enumerate(self.train_loader):
                i += 1
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
                        epoch+1, i, total_step, loss_fc.data[0]))

                # evaluation with test dataset (800 per class)
                if (i % len(self.train_loader) == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    for im, la in self.test_loader:
                        # volatile means this Variable requires no grad computation
                        im_test = Variable(im, volatile=True).cuda()
                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        for index in range(la.size(0)):
                            label_mask[index, la[index]] = 1
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        outputs = self.cnn(img_test_resized.detach())
                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / la.size(0)
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                    correct_meanscore /= 25  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.cnn.state_dict(), './data/best-pre_resnet50.pth')

    def train_adversarial(self, opt):
        best_score=0

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(100):
            self.discriminator.train()
            self.decoder.train()
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################
                i += 1
                labels_real = Variable(torch.ones(images[0].size(0)).fill_(self.real_label)).long()
                labels_fake = Variable(torch.zeros(images[0].size(0)).fill_(self.fake_label)).long()

                image_class = Variable(images[1].cuda())
                cls0_mask = Variable(torch.zeros(images[0].size(0))).cuda().long()
                for index in range(images[0].size(0)):
                    if image_class.data[index] == self.cls:
                        cls0_mask[index] = 1

                images = images[0].cuda()
                labels_real = labels_real.cuda()
                labels_fake = labels_fake.cuda()

                images = Variable(images)
                images_resized = func.upsample_bilinear(images, (224, 224))


                self.decoder.zero_grad()
                self.discriminator.zero_grad()
                self.imitator.fc.zero_grad()

                # Train discriminator with real image
                mask = self.decoder(self.encoder(images_resized))

                # combined image
                image_result = images_resized.detach() + mask #- (images_resized.detach())*mask

                # mask = mask * 0.5  # mask *= 0.01 is inplace operation(cannot compute gradient)
                logit_real = self.discriminator(images_resized.detach())
                loss_real_real = self.criterion_D(logit_real, cls0_mask)

                #backward the discriminator
                loss_discriminator = loss_real_real  # + loss_fake_fake
                loss_discriminator.backward()
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.imitator.fc.zero_grad()
                #training the imitator -> imitate the gt result + blackbox_classifier result
                labels_img_cls_true = func.softmax(self.cnn(images_resized.detach()))
                # labels_img_cls_fake = func.softmax(self.cnn(image_result.detach()))

                labels_img_imit_true = func.softmax(self.imitator(images_resized.detach()))
                # labels_img_imit_fake = func.softmax(self.imitator(image_result.detach()))

                # loss for the imitator: note that the cls network is independent to the imitator
                loss_imitator_true = self.criterion_I(labels_img_imit_true, labels_img_cls_true.detach())
                # loss_imitator_fake = self.criterion_I(labels_img_imit_fake, labels_img_cls_fake.detach())

                loss_imitator = loss_imitator_true
                # loss_imitator = loss_imitator_fake + loss_imitator_true
                loss_imitator.backward()
                clip_gradient(self.optim_I, 0.5)
                self.optim_I.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.imitator.fc.zero_grad()

                mask = self.decoder(self.encoder(images_resized))
                image_result = images_resized.detach() + mask
                logit_fake = self.discriminator(image_result)
                loss_fake_real = self.criterion_D(logit_fake, labels_real)

                #l1 regularization
                image_l1 = mask + images_resized
                loss_l1 = self.criterion_L1(image_l1, images_resized)

                #adversarial classification - use imitator instead of the cls
                cnn_out = self.imitator(image_result)
                label_target = Variable(self.cls*torch.ones(images.size(0))).cuda().long()

                loss_cls = self.criterion_G_CNN(cnn_out, label_target.detach())

                #accumulated error
                if i % 50 == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_real_real: %.4f, loss_fake_real: %.4f, loss_cls: %.4f'
                          % (epoch + 1, epoch, i, total_step,loss_real_real.data[0], loss_fake_real.data[0], loss_cls.data[0]))

                #backward the generator
                loss_generator = loss_fake_real + 100*loss_l1 + loss_cls  #initially we set weights as 1
                loss_generator.backward()
                clip_gradient(self.optim_G, 0.5)
                self.optim_G_dis.step()

                # Test the Model
                if (i % len(self.train_loader) == 0) and (i != 0) and(epoch % 2 == 0):
                    total = 0
                    correct = 0
                    correct_meanscore = 0
                    correct_i = 0
                    correct_meanscore_i = 0

                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_target = self.cls * torch.ones(la.size(0)).long()
                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        for index in range(la.size(0)):
                            label_mask[index, self.cls] = 1
                        label_mask_i = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        for index in range(la.size(0)):
                            label_mask_i[index, la[index]] = 1
                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = img_test_resized + mask_test #- img_test_resized*mask_test
                        outputs = self.imitator(reconst_images)
                        output_i = self.imitator(img_test_resized)

                        _, predicted = torch.max(outputs.data, 1)
                        _, predicted_i = torch.max(output_i.data, 1)

                        total += la.size(0)

                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / la.size(0)
                        correct_meanscore += c
                        correct += (predicted.cpu() == label_target).sum()

                        a_i = func.softmax(output_i)
                        b_i = a_i*label_mask_i
                        c_i = torch.sum(b_i) / la.size(0)
                        correct_meanscore_i += c_i
                        correct_i += (predicted_i.cpu() == la).sum()

                        if j % 100 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= (500/self.batch_size)
                    correct_meanscore_i /= (500/self.batch_size)

                    print('Test Accuracy of the imitator on the masked images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    print('Test Accuracy of the imitator on the original images: %d %%' % (100 * correct_i / total))
                    print('Mean Accuracy of the imitator: %.4f' % correct_meanscore_i.data[0])

                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.decoder.state_dict(), './data/best-generator.pth')
                        torch.save(self.discriminator.state_dict(), './data/best-discriminator.pth')
                        torch.save(self.optim_G_dis.state_dict(), './data/best-optimizer.pth')
                if (i % len(self.train_loader) == 0) and (i != 0) and (epoch % 10 == 0):
                    total = 0
                    correct = 0
                    correct_meanscore = 0

                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_target = self.cls * torch.ones(la.size(0)).long()
                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        for index in range(la.size(0)):
                            label_mask[index, self.cls] = 1
                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = img_test_resized + mask_test #- img_test_resized*mask_test
                        outputs = self.cnn_2(reconst_images)

                        _, predicted = torch.max(outputs.data, 1)

                        total += la.size(0)

                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / la.size(0)
                        correct_meanscore += c
                        correct += (predicted.cpu() == label_target).sum()

                    correct_meanscore /= (500/self.batch_size)

                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

    def evaluation(self):
        correct = 0
        total = 0
        correct_meanscore = 0
        j = 0
        for im, la in self.test_loader:
            j += 1
            im_test = Variable(im, volatile=True).cuda()
            img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
            label_target = self.cls * torch.ones(la.size(0)).long()
            label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
            for index in range(la.size(0)):
                label_mask[index, self.cls] = 1
            mask_test = self.decoder(self.encoder(img_test_resized))
            reconst_images = img_test_resized + mask_test  # - img_test_resized*mask_test
            outputs = self.cnn_2(reconst_images)

            _, predicted = torch.max(outputs.data, 1)
            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / la.size(0)
            correct_meanscore += c
            total += la.size(0)
            correct += (predicted.cpu() == label_target).sum()
            if j % 100 == 0:
                torchvision.utils.save_image(img_test_resized.data.cpu(),
                                             './data/test_images_%d.jpg' % (j))
                torchvision.utils.save_image(mask_test.data.cpu(),
                                             './data/test_noise_%d.jpg' % (j))
                torchvision.utils.save_image(reconst_images.data.cpu(),
                                             './data/test_reconst_images_%d.jpg' % (j))
        correct_meanscore /= (8000/self.batch_size)
        print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
