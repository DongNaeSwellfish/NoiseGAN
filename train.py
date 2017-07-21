import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Encoder, Decoder, ConvMaskGenerator
import torchvision.models as models
import torch.nn.functional as func


class Trainer(object):
    def __init__(self, trainloader, testloader, opt):
        self.opt = opt
        self.num_gpu = 1
        self.batch_size = 100

        self.train_loader = trainloader
        self.test_loader = testloader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(conv_dim=64).cuda()
        self.conv_gen = ConvMaskGenerator(conv_dim=64).cuda()

        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()
        self.cnn = models.resnet50(pretrained=True)
        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 1000)
        self.finetune(allow=True)

        # self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        # self.optim_G_dis = optim.Adam(self.decoder.parameters(), lr=0.001)
        # self.optim_G_cls = optim.Adam(self.decoder.parameters(), lr=0.001)
        # self.optim_L1 = optim.Adam(self.decoder.parameters(), lr=0.001)
        # self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0005)

        # self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        self.optim_G_dis_conv = optim.Adam(self.conv_gen.parameters(), lr=0.002)
        self.optim_G_cls_conv = optim.Adam(self.conv_gen.parameters(), lr=0.001)
        self.optim_L1_conv = optim.Adam(self.conv_gen.parameters(), lr=0.0005)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0005)

        # self.criterion_C = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.MSELoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()

        self.real_label = 1
        self.fake_label = 0

        # if torch.cuda.is_available():
        #     # self.encoder.cuda()
        #     # self.decoder.cuda()
        #     self.discriminator.cuda()
        #     self.cnn.cuda()
        #     self.cnn.fc.cuda()
        #     self.conv_gen.cuda()

    # if allow = True, classifier resnet50 computes grad
    def finetune(self, allow=True):
        for param in self.cnn.parameters():
            param.requires_grad = False
        # for param in self.cnn.fc.parameters():
        #     param.requires_grad = True if allow else False

    # Train the fully-connected layer of resnet50 with STL10 dataset
    def train_classifier(self, opt):
        if opt.workers > 1:
            self.cnn = torch.nn.DataParallel(self.cnn.cuda(), device_ids=range(3))
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)

        for epoch in range(1):
            # for i, images in enumerate(self.train_loader):
            #     self.cnn.zero_grad()
            #     images_label = Variable(images[1]).long().cuda()
            #     images = images[0].float().cuda()
            #     images = Variable(images)
            #
            #     img_resized = func.upsample_bilinear(images, size=(224, 224))  # (96x96 -> 224x224)
            #
            #     cnn_out = self.cnn(img_resized.detach())
            #     _, gfwe = torch.max(cnn_out.data, 1)
            #     loss_fc = self.criterion_C(cnn_out, images_label)
            #     loss_fc.backward()  # make graph
            #     clip_gradient(self.optim_C, 0.5)
            #     self.optim_C.step()  # update with gradient
            #
            #     if (i % 10) == 0:
            #         print('Epoch [%d/5], Step[%d/%d], classification loss: %.4f, ' % (
            #             epoch + 1, i, total_step, loss_fc.data[0]))

                # evaluation with test dataset (800 per class)
                # if (i % (len(self.train_loader)-1) == 0) and (i != 0):
            # if i == 3000:
            correct = 0
            total = 0
            correct_meanscore = 0
            for im, la in self.test_loader:
                # volatile means this Variable requires no grad computation
                im_test = Variable(im, volatile=True).cuda()
                label_mask = Variable(torch.zeros(self.batch_size, 1000), volatile=True).cuda()
                for index in range(self.batch_size):
                    label_mask[index, la[0][index]-1] = 1
                img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                outputs = self.cnn(img_test_resized.detach())
                _, predicted = torch.max(outputs.data, 1)
                a = func.softmax(outputs)
                b = a * label_mask
                c = torch.sum(b) / self.batch_size
                correct_meanscore += c
                total += im.size(0)
                correct += (predicted.cpu() == (la[0]-1)).sum()
            correct_meanscore /= len(self.test_loader)  # 200 = number of iteration in one test epoch
            print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
            print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
            break

    def train_conv_mask(self, opt):
        if opt.workers > 1:
            self.conv_gen = torch.nn.DataParallel(self.conv_gen.cuda(), device_ids=range(3))
            self.discriminator = torch.nn.DataParallel(self.discriminator.cuda(), device_ids=range(3))
            self.cnn = torch.nn.DataParallel(self.cnn.cuda(), device_ids=range(3))
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(200):
            self.discriminator.train()
            self.conv_gen.train()
            for i, images in enumerate(self.train_loader):
                i += 1
                labels_real = Variable(torch.zeros(len(images[1])).fill_(self.real_label)).long()
                labels_fake = Variable(torch.ones(len(images[1])).fill_(self.fake_label)).long()
                image_class = images[1].cuda()
                images = images[0].cuda()
                labels_real = labels_real.cuda()
                labels_fake = labels_fake.cuda()
                images = Variable(images)
                images_resized = func.upsample_bilinear(images, (224, 224))

                self.conv_gen.zero_grad()
                self.discriminator.zero_grad()
                self.cnn.zero_grad()

                ######################################################
                #                train Discriminator                 #
                ######################################################

                logit_real = self.discriminator(images_resized.detach())
                loss_real_real = self.criterion_D(logit_real, labels_real)
                loss_real_real.backward()
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                self.discriminator.zero_grad()
                self.conv_gen.zero_grad()

                conv_mask = self.conv_gen(images_resized)
                conv_masked_img = []
                for batch in range(images.size(0)):
                    conv_masked_img.append(
                        func.conv2d(images_resized[batch].unsqueeze(0), conv_mask[batch].view(3, 3, 7, 7), padding=3))
                conv_masked_img = torch.cat(conv_masked_img, dim=0)

                logit_fake = self.discriminator(conv_masked_img)
                loss_fake_fake = self.criterion_D(logit_fake, labels_fake)
                loss_fake_fake.backward(retain_variables=True)
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.cnn.zero_grad()
                self.conv_gen.zero_grad()

                conv_mask = self.conv_gen(images_resized)
                conv_masked_img = []
                for batch in range(images.size(0)):
                    conv_masked_img.append(
                        func.conv2d(images_resized[batch].unsqueeze(0), conv_mask[batch].view(3, 3, 7, 7), padding=3))
                conv_masked_img = torch.cat(conv_masked_img, dim=0)

                logit_fake = self.discriminator(conv_masked_img)
                loss_fake_real = self.criterion_D(logit_fake, labels_real)
                loss_fake_real.backward()  # (retain_variables=True)
                clip_gradient(self.optim_G_dis_conv, 0.5)
                self.optim_G_dis_conv.step()

                self.discriminator.zero_grad()
                self.conv_gen.zero_grad()
                # self.cnn.fc.zero_grad()

                conv_mask = self.conv_gen(images_resized)
                conv_masked_img = []
                for batch in range(images.size(0)):
                    conv_masked_img.append(
                        func.conv2d(images_resized[batch].unsqueeze(0), conv_mask[batch].view(3, 3, 7, 7), padding=3))
                conv_masked_img = torch.cat(conv_masked_img, dim=0)

                loss_l1 = self.criterion_L1(conv_masked_img, images_resized)
                loss_l1 = loss_l1 * 100
                loss_l1.backward()
                self.optim_L1_conv.step()

                self.discriminator.zero_grad()
                self.cnn.zero_grad()
                self.conv_gen.zero_grad()

                # Train generator with fake image with classifier (resnet50)
                conv_mask = self.conv_gen(images_resized)
                conv_masked_img = []
                for batch in range(images.size(0)):
                    conv_masked_img.append(
                        func.conv2d(images_resized[batch].unsqueeze(0), conv_mask[batch].view(3, 3, 7, 7), padding=3))
                conv_masked_img = torch.cat(conv_masked_img, dim=0)

                cnn_out = self.cnn(conv_masked_img)
                cnn_out = func.softmax(cnn_out)
                label_target = Variable(torch.ones((images.size(0), 1))).cuda()
                # MSE loss with uniform distribution of 0.5
                one_mask = Variable(torch.zeros(images.size(0), 1000)).cuda()
                for index in range(images.size(0)):
                    one_mask[index, image_class[index]] = 1
                cnn_out_truth = one_mask.detach() * cnn_out
                cnn_out_truth = torch.sum(cnn_out_truth, dim=1)
                loss_cls = self.criterion_G_CNN(cnn_out_truth, label_target.detach())
                loss_cls.backward()
                clip_gradient(self.optim_G_cls_conv, 0.5)
                self.optim_G_cls_conv.step()

                if (i % 10) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_fake_real: %.4f,'
                          ' ''loss_real_real: %.4f, loss_fake_fake: %.4f, cls_loss: %.4f, l1_loss: %.4f'
                          % (epoch + 1, epoch, i + 1, total_step, loss_fake_real.data[0], loss_real_real.data[0],
                             loss_fake_fake.data[0], loss_cls.data[0], loss_l1.data[0]))

                # Test the Model
                if (i % (len(self.train_loader)) == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        # if j == 500:
                        #     break
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_mask = Variable(torch.zeros(im.size(0), 1000), volatile=True).cuda()
                        for index in range(im.size(0)):
                            label_mask[index, (la[0])[index]-1] = 1
                        conv_mask_test = self.conv_gen(img_test_resized)
                        conv_masked_img = []
                        for batch in range(im.size(0)):
                            conv_masked_img.append(
                                func.conv2d(img_test_resized[batch].unsqueeze(0), conv_mask_test[batch].view(3, 3, 7, 7),
                                            padding=3))
                        conv_masked_img = torch.cat(conv_masked_img, dim=0)
                        outputs = self.cnn(conv_masked_img)

                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / im.size(0)
                        correct_meanscore += c
                        total += self.batch_size
                        correct += (predicted.cpu() == la[0]-1).sum()
                        if j % 100 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            # torchvision.utils.save_image(conv_mask_test.data.cpu(),
                            #                              './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(conv_masked_img.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= len(self.test_loader)
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

    def train_adversarial(self, opt):
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(200):
            self.discriminator.train()
            self.decoder.train()
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

                self.decoder.zero_grad()
                self.discriminator.zero_grad()

                # Train discriminator with real image
                mask = self.decoder(self.encoder(images_resized))
                # mask = mask * 0.5  # mask *= 0.01 is inplace operation(cannot compute gradient)
                logit_real = self.discriminator(images_resized.detach())
                loss_real_real = self.criterion_D(logit_real, labels_real)
                loss_real_real.backward()
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                self.discriminator.zero_grad()
                self.decoder.zero_grad()

                # Train discriminator with fake image
                logit_fake = self.discriminator(images_resized.detach()+mask)
                loss_fake_fake = self.criterion_D(logit_fake, labels_fake)
                loss_fake_fake.backward(retain_variables=True)

                # if (i % 2) == 0:
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.cnn.fc.zero_grad()
                logit_fake = self.discriminator(images_resized.detach() + mask)
                loss_fake_real = self.criterion_D(logit_fake, labels_real)
                loss_fake_real.backward()  # (retain_variables=True)
                clip_gradient(self.optim_G_dis, 0.5)
                self.optim_G_dis.step()

                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.cnn.fc.zero_grad()
                mask = self.decoder(self.encoder(images_resized))
                image_l1 = mask + images_resized
                loss_l1 = self.criterion_L1(image_l1, images_resized)
                loss_l1 = loss_l1 * 500
                loss_l1.backward()
                self.optim_L1.step()

                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.cnn.fc.zero_grad()

                # Train generator with fake image with classifier (resnet50)
                mask = self.decoder(self.encoder(images_resized))
                # mask = mask * 0.5
                cnn_out = self.cnn(images_resized.detach()+mask)
                cnn_out = func.softmax(cnn_out)

                label_target = Variable(torch.zeros((self.batch_size, 1))).cuda()
                # MSE loss with uniform distribution of 0.5
                one_mask = Variable(torch.zeros(self.batch_size, 1000)).cuda()
                for index in range(self.batch_size):
                    one_mask[index, image_class[index]] = 1
                cnn_out_truth = one_mask.detach() * cnn_out
                cnn_out_truth = torch.sum(cnn_out_truth, dim=1)
                loss_cls = self.criterion_G_CNN(cnn_out_truth, label_target.detach())
                loss_cls.backward()
                clip_gradient(self.optim_G_cls, 0.5)
                self.optim_G_cls.step()

                if (i % 10) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_fake_real: %.4f,'
                          ' ''loss_real_real: %.4f, loss_fake_fake: %.4f, cls_loss: %.4f, l1_loss: %.4f'
                          % (epoch + 1, epoch, i + 1, total_step, loss_fake_real.data[0], loss_real_real.data[0],
                             loss_fake_fake.data[0], loss_cls.data[0], loss_l1.data[0]))

                # Test the Model
                if (i % 249 == 0) and (i != 0):
                    correct = 0
                    total = 0
                    correct_meanscore = 0
                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))
                        label_mask = Variable(torch.zeros(self.batch_size, 1000), volatile=True).cuda()
                        for index in range(self.batch_size):
                            label_mask[index, la[index]] = 1
                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = img_test_resized + mask_test
                        outputs = self.cnn(reconst_images)

                        _, predicted = torch.max(outputs.data, 1)
                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / self.batch_size
                        correct_meanscore += c
                        total += la.size(0)
                        correct += (predicted.cpu() == la).sum()
                        if j % 100 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= 400
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
