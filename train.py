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


class Trainer(object):
    def __init__(self, trainloader, testloader, opt):
        self.opt = opt
        self.num_gpu = 1
        self.batch_size = 10

        self.train_loader = trainloader
        self.test_loader = testloader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(conv_dim=64).cuda()

        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()

        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)

        self.pre_cnn_path = os.path.join('/home', 'choikoal', 'NoiseGAN_imgnet_stloss', 'data', 'best-pre_resnet50.pth')
        self.cnn.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained model from %s' % self.pre_cnn_path)

        self.cnn_2 = models.resnet101(pretrained=True)
        self.cnn_2.fc = nn.Linear(self.cnn_2.fc.in_features, 10)

        self.pre_cnn_2_path = os.path.join('/home', 'choikoal', 'NoiseGAN_imgnet_stloss', 'data', 'best-pre_resnet101.pth')
        self.cnn_2.load_state_dict(torch.load(self.pre_cnn_2_path))
        print('load pretrained model from %s' % self.pre_cnn_2_path)

        #define imitator
        self.imitator = models.resnet34(pretrained=True)
        self.imitator.fc = nn.Linear(self.imitator.fc.in_features,10)


        self.finetune(allow=True)

        self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        self.optim_G_dis = optim.Adam(self.decoder.parameters(), lr=0.0005)
        self.optim_G_cls = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_G = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.optim_L1 = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_I = optim.Adam(self.imitator.fc.parameters(), lr = 0.001)

        self.criterion_C = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.MSELoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()
        self.criterion_I = nn.SmoothL1Loss()

        self.real_label = 1
        self.fake_label = 0

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
            param.requires_grad = True
        for param in self.cnn.fc.parameters():
            param.requires_grad = True if allow else False

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

                if (i % 50) == 0:
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
                    correct_meanscore /= 50  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.cnn.state_dict(), './data/best-pre_resnet.pth')

    def train_adversarial(self, opt):
        best_score = 1

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(5):
            self.discriminator.train()
            self.decoder.train()
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################
                i += 1
                labels_real = Variable(torch.zeros(images[0].size(0)).fill_(self.real_label)).long()
                labels_fake = Variable(torch.ones(images[0].size(0)).fill_(self.fake_label)).long()

                image_class = images[1].cuda()
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

                disc_gt = self.discriminator(images_resized.detach())
                disc = self.discriminator(image_result)

                #loss real
                logit_real = disc_gt[0]
                loss_real_real = self.criterion_D(logit_real, labels_real)
                logit_fake = disc[0]
                loss_fake_fake = self.criterion_D(logit_fake, labels_fake)

                #backward the discriminator
                loss_discriminator = loss_real_real + loss_fake_fake
                loss_discriminator.backward(retain_variables=True)
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()


                #training the imitator -> imitate the gt result + blackbox_classifier result
                labels_img_cls_true = func.softmax(self.cnn(images_resized.detach()))
                labels_img_cls_fake = func.softmax(self.cnn(image_result.detach()))

                labels_img_imit_true = func.softmax(self.imitator(images_resized.detach()))
                labels_img_imit_fake = func.softmax(self.imitator(image_result.detach()))

                # loss for the imitator: note that the cls network is independent to the imitator
                loss_imitator_true = self.criterion_I(labels_img_imit_true, labels_img_cls_true.detach())
                loss_imitator_fake = self.criterion_I(labels_img_imit_fake, labels_img_cls_fake.detach())

                loss_imitator = loss_imitator_fake + loss_imitator_true
                loss_imitator.backward(retain_variables = True)
                clip_gradient(self.optim_I, 0.5)
                self.optim_I.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                self.cnn.fc.zero_grad()

                #generator loss
                logit_fake = disc[0]
                loss_fake_real = self.criterion_D(logit_fake, labels_real)

                # style loss
                feat_1 = disc[1].detach()
                # feat_1 = feat_1.view([self.batch_size,128, 12544])
                # gram_1 = torch.matmul(feat_1, torch.transpose(feat_1,1,2))

                # texture_loss
                feat_1 = feat_1.view([images.size(0), 128, 49, 256])
                feat_1 = torch.transpose(feat_1, 0, 3)
                gram_1 = 0
                for j in range(0, 256):
                    gram_1_i = torch.matmul(feat_1[j], torch.transpose(feat_1[j], 1, 2))
                    gram_1 += gram_1_i

                feat_2 = disc[2].detach()
                # feat_2 = feat_2.view([self.batch_size, 256, 3136])
                # gram_2 = torch.matmul(feat_2, torch.transpose(feat_2, 1, 2))

                # tloss
                feat_2 = feat_2.view([images.size(0), 256, 49, 64])
                feat_2 = torch.transpose(feat_2, 0, 3)
                gram_2 = 0
                for j in range(0, 64):
                    gram_2_i = torch.matmul(feat_2[j], torch.transpose(feat_2[j], 1, 2))
                    gram_2 += gram_2_i

                feat_3 = disc[3].detach()
                # feat_3 = feat_3.view([self.batch_size, 512, 784])
                # gram_3 = torch.matmul(feat_3, torch.transpose(feat_3, 1, 2))

                # tloss
                feat_3 = feat_3.view([images.size(0), 512, 49, 16])
                feat_3 = torch.transpose(feat_3, 0, 3)
                gram_3 = 0
                for j in range(0, 16):
                    gram_3_i = torch.matmul(feat_3[j], torch.transpose(feat_3[j], 1, 2))
                    gram_3 += gram_3_i

                feat_1_gt = disc_gt[1].detach()
                # feat_1_gt = feat_1_gt.view([self.batch_size, 128, 12544])
                # gram_1_gt = torch.matmul(feat_1_gt, torch.transpose(feat_1_gt, 1, 2))

                feat_1_gt = feat_1_gt.view([images.size(0), 128, 49, 256])
                feat_1_gt = torch.transpose(feat_1_gt, 0, 3)
                gram_1_gt = 0
                for j in range(0, 256):
                    gram_1_gt_i = torch.matmul(feat_1_gt[j], torch.transpose(feat_1_gt[j], 1, 2))
                    gram_1_gt += gram_1_gt_i

                feat_2_gt = disc_gt[2].detach()
                # feat_2_gt = feat_2_gt.view([self.batch_size, 256, 3136])
                # gram_2_gt = torch.matmul(feat_2_gt, torch.transpose(feat_2_gt, 1, 2))

                feat_2_gt = feat_2_gt.view([images.size(0), 256, 49, 64])
                feat_2_gt = torch.transpose(feat_2_gt, 0, 3)
                gram_2_gt = 0
                for j in range(0, 64):
                    gram_2_gt_i = torch.matmul(feat_2_gt[j], torch.transpose(feat_2_gt[j], 1, 2))
                    gram_2_gt += gram_2_gt_i

                feat_3_gt = disc_gt[3].detach()
                # feat_3_gt = feat_3_gt.view([self.batch_size, 512, 784])
                # gram_3_gt = torch.matmul(feat_3_gt, torch.transpose(feat_3_gt, 1, 2))

                feat_3_gt = feat_3_gt.view([images.size(0), 512, 49, 16])
                feat_3_gt = torch.transpose(feat_3_gt, 0, 3)
                gram_3_gt = 0
                for j in range(0, 16):
                    gram_3_gt_i = torch.matmul(feat_3_gt[j], torch.transpose(feat_3_gt[j], 1, 2))
                    gram_3_gt += gram_3_gt_i

                style_loss_1 = torch.norm(gram_1 - gram_1_gt)
                style_loss_2 = torch.norm(gram_2 - gram_2_gt)
                style_loss_3 = torch.norm(gram_3 - gram_3_gt)

                # # tensorboard
                # info = {
                #     'style_loss_1': style_loss_1.data[0] * 3e-7,
                #     'style_loss_2': style_loss_2.data[0] * 1e-6,
                #     'style_loss_3': style_loss_3.data[0] * 1e-6
                # }
                #
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, 250 * epoch + i)
                #

                #l1 regularization
                image_l1 = mask + images_resized
                loss_l1 = self.criterion_L1(image_l1, images_resized)
                #loss_l1_v = torch.nn.functional.relu(loss_l1 - self.l1_param) \
                #            + torch.nn.functional.relu(self.l1_param - loss_l1)

                #adversarial classification - use imitator instead of the cls
                cnn_out = self.imitator(image_result)
                cnn_out = func.softmax(cnn_out)
                label_target = Variable(torch.zeros((images.size(0), 1))).cuda()
                one_mask = Variable(torch.zeros(images.size(0), 10)).cuda()
                for index in range(images.size(0)):
                    one_mask[index, image_class[index]] = 1

                cnn_out_truth = one_mask.detach() * cnn_out
                cnn_out_truth = torch.sum(cnn_out_truth, dim=1)
                loss_cls = self.criterion_G_CNN(cnn_out_truth, label_target.detach())

                #accumulated error
                #loss_generator = sum(loss_fake_real, loss_l1, loss_cls)


                #backward the generator
                loss_generator = loss_fake_real + 5*loss_l1 + loss_cls + style_loss_1*3*1e-6 + style_loss_2*1*1e-5 + style_loss_3*1e-5
                #initially we set weights as 1
                loss_generator.backward()
                clip_gradient(self.optim_G, 0.5)
                self.optim_G_dis.step()

                if (i % 50) == 0:
                    print('Epoch [%d/%d], Step[%d/%d], loss_fake_real: %.4f,'
                          ' ''loss_real_real: %.4f, loss_fake_fake: %.4f, cls_loss: %.4f, l1_loss: %.4f, style_loss: %.4f, style_loss_2: %.4f, style_loss_3: %.4f'
                          % (epoch + 1, epoch, i, total_step, loss_fake_real.data[0], loss_real_real.data[0],
                             loss_fake_fake.data[0], loss_cls.data[0], loss_l1.data[0],
                             style_loss_1.data[0] * 3e-6, style_loss_2.data[0] * 1e-5, style_loss_3.data[0] * 1e-5))

                # Test the Model
                if (i % len(self.train_loader) == 0) and (i != 0):
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
                        label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
                        for index in range(self.batch_size):
                            label_mask[index, la[index]] = 1
                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = img_test_resized + mask_test  # - img_test_resized*mask_test
                        outputs = self.cnn(reconst_images)
                        output_i = self.imitator(img_test_resized)

                        _, predicted = torch.max(outputs.data, 1)
                        _, predicted_i = torch.max(output_i.data, 1)

                        total += la.size(0)

                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / self.batch_size
                        correct_meanscore += c
                        correct += (predicted.cpu() == la).sum()

                        a_i = func.softmax(output_i)
                        b_i = a_i * label_mask
                        c_i = torch.sum(b_i) / self.batch_size
                        correct_meanscore_i += c_i
                        correct_i += (predicted_i.cpu() == la).sum()

                        if j % 10 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= 50
                    correct_meanscore_i /= 50

                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    print('Test Accuracy of the imitator on the test images: %d %%' % (100 * correct_i / total))
                    print('Mean Accuracy of the imitator: %.4f' % correct_meanscore_i.data[0])

                    if correct_meanscore.data[0] < best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.decoder.state_dict(), './data/best-generator.pth')
                        torch.save(self.discriminator.state_dict(), './data/best-discriminator.pth')
                        torch.save(self.optim_G_dis.state_dict(), './data/best-optimizer.pth')

    def evaluation(self, opt):
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
            label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
            for index in range(self.batch_size):
                label_mask[index, la[index]] = 1
            mask_test = self.decoder(self.encoder(img_test_resized))
            reconst_images = img_test_resized + mask_test  # - img_test_resized*mask_test
            outputs = self.cnn_2(reconst_images)
            output_i = self.imitator(img_test_resized)

            _, predicted = torch.max(outputs.data, 1)
            _, predicted_i = torch.max(output_i.data, 1)

            total += la.size(0)

            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / self.batch_size
            correct_meanscore += c
            correct += (predicted.cpu() == la).sum()

            a_i = func.softmax(output_i)
            b_i = a_i * label_mask
            c_i = torch.sum(b_i) / self.batch_size
            correct_meanscore_i += c_i
            correct_i += (predicted_i.cpu() == la).sum()

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
        print('Test Accuracy of the imitator on the test images: %d %%' % (100 * correct_i / total))
        print('Mean Accuracy of the imitator: %.4f' % correct_meanscore_i.data[0])
