import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from model import Discriminator_cls
from model import Unet_Generator
import torchvision.models as models
import torch.nn.functional as func
import os


class Trainer(object):
    def __init__(self, trainloader, testloader):
        self.num_gpu = 1
        self.batch_size = 40

        self.train_loader = trainloader
        self.test_loader = testloader

        self.discriminator_cls = Discriminator_cls(image_size=224, conv_dim=64).cuda()
        self.unet = Unet_Generator(conv_dim=64).cuda()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)
        self.pre_cnn_path = os.path.join('/home', 'david', 'NoiseGAN', 'data', 'best-pre_resnet.pth')
        self.cnn.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained CNN model from %s' % self.pre_cnn_path)

        # self.pre_gen_path = os.path.join('/home', 'david', 'NoiseGAN', 'data', 'cls0-500-82', 'best-generator.pth')
        # self.decoder.load_state_dict(torch.load(self.pre_gen_path))
        # print('load pretrained generator model from %s' % self.pre_gen_path)

        self.finetune(allow=True)

        self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        self.optim_G_dis = optim.Adam(self.unet.parameters(), lr=0.002)
        self.optim_D = optim.Adam(self.discriminator_cls.parameters(), lr=0.0002)
        self.optim_L1 = optim.Adam(self.unet.parameters(), lr=0.002)

        self.criterion_C = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()
        self.criterion_D_cls = nn.CrossEntropyLoss()
        self.real_label = 1
        self.fake_label = 0
        self.cls = 0

        if torch.cuda.is_available():
            self.unet.cuda()
            self.cnn.cuda()
            self.cnn.fc.cuda()

    # if allow = True, classifier resnet50 computes grad
    def finetune(self, allow=True):
        for param in self.cnn.parameters():
            param.requires_grad = True
        for param in self.cnn.fc.parameters():
            param.requires_grad = True if allow else False

    # Train the fully-connected layer of resnet50 with STL10 dataset
    def train_classifier(self):
        best_score=0
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(10):
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
                        epoch + 1, i, total_step, loss_fc.data[0]))

                # evaluation with test dataset (800 per class)
                if (i % len(self.train_loader) == 0) and (i != 0):
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
                    correct_meanscore /= 200  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.cnn.state_dict(), './data/best-pre_resnet101.pth')

    def train_adversarial(self):
        best_score = 0

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

        total_step = len(self.train_loader)
        for epoch in range(500):
            self.unet.train()
            self.discriminator_cls.train()
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################
                self.discriminator_cls.zero_grad()
                self.unet.zero_grad()
                self.cnn.zero_grad()

                i += 1
                labels_real = Variable(torch.zeros(self.batch_size).fill_(self.real_label)).long()
                labels_fake = Variable(torch.ones(self.batch_size).fill_(self.fake_label)).long()

                image_class = Variable(images[1].cuda())
                cls0_mask = Variable(torch.zeros(self.batch_size)).cuda().long()
                for index in range(self.batch_size):
                    if image_class.data[index] == self.cls:
                        cls0_mask[index] = 1

                images = images[0].cuda()
                labels_real = labels_real.cuda()
                labels_fake = labels_fake.cuda()
                images = Variable(images)
                images_resized = func.upsample_bilinear(images, (224, 224))

                # Train discriminator with real image
                logit_cls0_real, _, _, _, _ = self.discriminator_cls(images_resized.detach())
                loss_cls0_real = self.criterion_D_cls(logit_cls0_real, cls0_mask)

                # backward the discriminator
                loss_discriminator = loss_cls0_real
                loss_discriminator.backward()
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator_cls.zero_grad()
                self.unet.zero_grad()
                self.cnn.zero_grad()
                _, res1, res2, res3, res4 = self.discriminator_cls(images_resized.detach())
                mask = self.unet(images_resized, res1.detach(), res2.detach(), res3.detach(), res4.detach())
                image_result = images_resized.detach() + mask

                # loss_l1 = self.criterion_L1(image_result, images_resized)

                logit_cls0_fake, _, _, _, _ = self.discriminator_cls(image_result.detach())
                loss_cls0_fake = self.criterion_D_cls(logit_cls0_fake, labels_real)

                loss_generator = loss_cls0_fake  # + 10 * loss_l1
                loss_generator.backward()
                # clip_gradient(self.optim_G_dis, 0.5)
                self.optim_G_dis.step()

                # if (i % 25) == 0:
                #     print('Epoch [%d/%d], Step[%d/%d], l1_loss: %.4f, loss_cls0_real: %.4f, loss_cls0_fake: %.4f'
                #           % (epoch + 1, epoch, i, total_step, loss_l1.data[0], loss_cls0_real.data[0], loss_cls0_fake.data[0]))
                if (i % 25) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_cls0_real: %.4f, loss_cls0_fake: %.4f'
                          % (epoch + 1, epoch, i, total_step, loss_cls0_real.data[0], loss_cls0_fake.data[0]))
                # # tensorboard_Scalar_logger
                # info = {
                #     'loss_fake_real': loss_fake_real.data[0],
                #     'loss_cls': loss_cls.data[0],
                #     'loss_fake_fake': loss_fake_fake.data[0]
                # }
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, i + 250 * epoch)

                # Test the Model
                if (i % len(self.train_loader) == 0) and (i != 0) and (epoch % 3 == 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_target = self.cls*torch.ones(la.size(0)).long()
                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        for index in range(la.size(0)):
                            label_mask[index, self.cls] = 1
                        _, res1_test, res2_test, res3_test, res4_test = self.discriminator_cls(img_test_resized.detach())
                        mask_test = self.unet(img_test_resized, res1_test, res2_test, res3_test, res4_test)
                        reconst_images = img_test_resized + mask_test
                        outputs = self.cnn(reconst_images)

                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask

                        c = torch.sum(b) / self.batch_size
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == label_target).sum()
                        if j % 100 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= 200
                    print predicted
                    print('Test Accuracy of the model on the test images for class 0 : %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.unet.state_dict(), './data/best-generator.pth')
                        torch.save(self.discriminator_cls.state_dict(), './data/best-discriminator.pth')
                        torch.save(self.optim_G_dis.state_dict(), './data/best-optimizer.pth')

    def evaluation(self):
        epoch=0
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
            reconst_images = img_test_resized + mask_test
            outputs = self.cnn(reconst_images)
            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
            torchvision.utils.save_image(mask_test.data.cpu(),
                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
            torchvision.utils.save_image(reconst_images.data.cpu(),
                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
            _, predicted = torch.max(outputs.data, 1)
            a = func.softmax(outputs)
            b = a * label_mask

            c = torch.sum(b) / self.batch_size
            correct_meanscore += c
            total += la.size(0)
            correct += (predicted.cpu() == label_target).sum()
        correct_meanscore /= 200
        print predicted
        print('Test Accuracy of the model on the test images for class 1 : %d %%' % (100 * correct / total))
        print('Mean Accuracy: %.4f' % correct_meanscore.data[0])