import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Discriminator_cls
from model import Encoder, Decoder
import torchvision.models as models
import torch.nn.functional as func
import os
from logger import Logger

logger = Logger('./logs')


class Trainer(object):
    def __init__(self, trainloader, testloader):
        self.num_gpu = 1
        self.batch_size = 20

        self.train_loader = trainloader
        self.test_loader = testloader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(conv_dim=64).cuda()

        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()
        self.discriminator_cls = Discriminator_cls(image_size=224, conv_dim=128).cuda()

        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)

        # positive bank
        self.p_bank = []
        self.maxP = 500

        # class bank
        self.c_bank = []
        self.maxC = 100

        # /home/yjyoo/Code/NoiseGAN-koal_stloss/data
        self.pre_cnn_path = os.path.join('/home', 'yjyoo', 'Code', 'NoiseGAN-imitation_1class', 'data',
                                         'best-pre_resnet101.pth')
        self.cnn.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained model from %s' % self.pre_cnn_path)

        self.cnn_2 = models.resnet50(pretrained=True)
        self.cnn_2.fc = nn.Linear(self.cnn_2.fc.in_features, 10)
        #
        self.pre_cnn_2_path = os.path.join('/home', 'yjyoo', 'Code', 'NoiseGAN-imitation_1class', 'data',
                                           'best-pre_resnet50.pth')
        self.cnn_2.load_state_dict(torch.load(self.pre_cnn_2_path))
        print('load pretrained model from %s' % self.pre_cnn_2_path)

        #define imitator
        #self.imitator = models.resnet34(pretrained=True)
        #self.imitator.fc = nn.Linear(self.imitator.fc.in_features,10)

        self.finetune(allow=True)

        #self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        self.optim_G_dis = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_G_cls = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.optim_G = optim.Adam(self.decoder.parameters(), lr = 0.0001)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.optim_D_cls = optim.Adam(self.discriminator_cls.parameters(), lr=0.0001)
        self.optim_L1 = optim.Adam(self.decoder.parameters(), lr=0.001)
        #self.optim_I = optim.Adam(self.imitator.fc.parameters(), lr = 0.001)

        self.criterion_C = nn.CrossEntropyLoss()
        #self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.CrossEntropyLoss()
        self.criterion_D_cls = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()
        #self.criterion_I = nn.SmoothL1Loss()


        self.real_label = 1
        self.fake_label = 0

        self.cls = 0

        self.l1_param = 0.001

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            self.discriminator_cls.cuda()
            self.cnn.cuda()
            self.cnn.fc.cuda()
            #self.imitator.cuda()
            #self.imitator.fc.cuda()
            self.cnn_2.cuda()
            self.cnn_2.fc.cuda()



    # if allow = True, classifier resnet50 computes grad
    def finetune(self, allow=True):
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.fc.parameters():
            param.requires_grad = False

    # Train the fully-connected layer of resnet50 with STL10 dataset
    def train_classifier(self):
        best_score = 0
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(5):
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
                    correct_meanscore /= 400  # 200 = number of iteration in one test epoch
                    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])

                    if correct_meanscore.data[0] > best_score:
                        best_score = correct_meanscore.data[0]
                        print("saving best model...")
                        torch.save(self.cnn.state_dict(), './data/best-pre_resnet.pth')

    def train_adversarial(self):
        num_pbank = 0
        num_cbank = 0

        best_score=0
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)
        for epoch in range(100):
            self.discriminator.train()
            self.decoder.train()
            cnt_tot = 0
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################
                i += 1
                labels_real = Variable(torch.ones(self.batch_size).fill_(self.real_label)).long()
                labels_fake = Variable(torch.zeros(self.batch_size).fill_(self.fake_label)).long()

                image_class = Variable(images[1].cuda())
                images = images[0].cuda()


                labels_real = labels_real.cuda()
                labels_fake = labels_fake.cuda()

                images = Variable(images)
                images_resized = func.upsample_bilinear(images, (224, 224))


                self.decoder.zero_grad()
                self.discriminator.zero_grad()
                self.discriminator_cls.zero_grad()
                #self.imitator.fc.zero_grad()

                # Train discriminator with real image
                mask = self.decoder(self.encoder(images_resized))

                # cls result of the combined image
                image_result = images_resized.detach() + mask
                _, cls_class = torch.max(func.softmax(self.cnn(image_result.detach())),1)

                #mask setting + bank stack
                cls0_mask = Variable(torch.zeros(self.batch_size)).cuda().long()
                cnt_pos = 0
                cnt_all = 0
                nZeroIdx = []
                nSelect = 0
                for index in range(self.batch_size):
                    #if image_class.data[index] == self.cls:
                    if cls_class.cpu().data.numpy()[index] == self.cls: # if real true

                        cls0_mask[index] = 1
                        cnt_all+=1
                        if image_class.cpu().data.numpy()[index] != self.cls:
                            cls0_mask[index] = 1
                            cnt_pos+=1
                            img_pos = image_result.cpu().data.numpy()[index].copy()
                            if num_pbank<self.maxP:
                                self.p_bank.append(img_pos.copy())
                                num_pbank+=1
                            else:
                                pidx = num_pbank%self.maxP
                                num_pbank += 1
                                self.p_bank[pidx] = img_pos.copy()
                    else:
                        nZeroIdx.append(index)

                image_batch = image_result.cpu().data.numpy()

                #renew the batch
                if cnt_pos < self.batch_size / 2:
                    selIdx = np.random.permutation(np.asarray(nZeroIdx))
                    bankIdx = np.random.permutation(min(num_pbank, self.maxP))
                    nSelect = max(min(self.batch_size / 2 - cnt_all, num_pbank), 0) #should be larger&eq than zero

                    for rev_id in range(nSelect):
                        image_batch[selIdx[rev_id]] = self.p_bank[bankIdx[rev_id]].copy()
                        cls0_mask[selIdx[rev_id]] = 1


                # generate class bank
                for index in range(self.batch_size):
                    if image_class.cpu().data.numpy()[index] == self.cls:

                        img_tgt = image_result.cpu().data.numpy()[index].copy()

                        if num_cbank < self.maxC:
                            self.c_bank.append(img_tgt.copy())
                            num_cbank+=1
                        else:
                            cidx = num_cbank % self.maxC
                            self.c_bank[cidx] = img_tgt.copy()
                            num_cbank += 1


                # mask = mask * 0.5  # mask *= 0.01 is inplace operation(cannot compute gradient)
                if cnt_pos + nSelect > 0 or cnt_all > 0: #train the network if there is at least one positive sample
                    image_batch_var = Variable(torch.FloatTensor(image_batch.copy())).cuda()
                    logit_real = self.discriminator(image_batch_var) #image_result because the logit value is the result from..
                    loss_real_real = self.criterion_D(logit_real[0], cls0_mask)

                    #backward the discriminator
                    loss_discriminator = loss_real_real  # + loss_fake_fake
                    loss_discriminator.backward()
                    clip_gradient(self.optim_D, 0.5)
                    self.optim_D.step()

                cnt_tot += cnt_pos


                # discriminator cls load
                if num_cbank > self.maxC:
                    selIdx = np.random.permutation(self.maxC)

                    for rev_id in range(self.batch_size):
                        image_batch[rev_id] = self.c_bank[selIdx[rev_id]].copy() #reuse old image batch

                    cimage_batch_var = Variable(torch.FloatTensor(image_batch.copy())).cuda()
                    # logit value for disc 2
                    clogit_real = self.discriminator_cls(cimage_batch_var)  # image_result because the logit value is the result from..
                    clogit_fake = self.discriminator_cls(image_result.detach())
                    closs_real_real = self.criterion_D_cls(clogit_real, labels_real)
                    closs_fake_fake = self.criterion_D_cls(clogit_fake, labels_fake)

                    # backward training
                    closs_discriminator = closs_real_real + closs_fake_fake
                    closs_discriminator.backward()
                    clip_gradient(self.optim_D_cls, 0.5)
                    self.optim_D_cls.step()




                #self.imitator.fc.zero_grad()
                #training the imitator -> imitate the gt result + blackbox_classifier result
                #labels_img_cls_true = func.softmax(self.cnn(images_resized.detach()))
                #labels_img_cls_fake = func.softmax(self.cnn(image_result.detach()))

                #labels_img_imit_true = func.softmax(self.imitator(images_resized.detach()))
                #labels_img_imit_fake = func.softmax(self.imitator(image_result.detach()))

                # loss for the imitator: note that the cls network is independent to the imitator
                #loss_imitator_true = self.criterion_I(labels_img_imit_true, labels_img_cls_true.detach())
                #loss_imitator_fake = self.criterion_I(labels_img_imit_fake, labels_img_cls_fake.detach())

                #loss_imitator = loss_imitator_true
                #loss_imitator = loss_imitator_fake + loss_imitator_true
                #loss_imitator.backward()
                #clip_gradient(self.optim_I, 0.5)
                #self.optim_I.step()

                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                #self.imitator.fc.zero_grad()

                mask = self.decoder(self.encoder(images_resized))

                if cnt_tot > 0: #if discriminator is trained at least once

                    image_result = images_resized.detach() + mask
                    # gan 1
                    logit_fake = self.discriminator(image_result)
                    loss_fake_real = self.criterion_D(logit_fake[0], labels_real) #all the labels to be true

                    #l1 regularization
                    image_l1 = mask + images_resized
                    loss_l1 = self.criterion_L1(image_l1, images_resized)

                    if num_cbank > self.maxC: # if discrminator 2 runs
                        #gan 2
                        clogit_fake_n = self.discriminator_cls(image_result)
                        closs_fake_real = self.criterion_D(clogit_fake_n, labels_real)
                        loss_generator = loss_fake_real + 500 * loss_l1 + closs_fake_real  # + loss_cls# + style_loss #initially we set weights as 1

                        if i % 50 == 0:
                            print(
                            'Epoch [%d/%d], Step[%d/%d], loss_real_real: %.4f,  closs_real_real: %.4f, loss_fake_real: %.4f, closs_fake_real: %.4f,  num cnt: %d'
                            % (epoch + 1, epoch, i, total_step, loss_real_real.data[0], closs_discriminator.data[0],
                               loss_fake_real.data[0], closs_fake_real.data[0], cnt_tot))
                            cnt_tot = 0
                    else:
                        loss_generator = loss_fake_real + 500 * loss_l1

                        if i % 50 == 0:
                            print('Epoch [%d/%d], Step[%d/%d], loss_real_real: %.4f, loss_fake_real: %.4f, num cnt: %d'
                                % (epoch + 1, epoch, i, total_step,loss_real_real.data[0], loss_fake_real.data[0],cnt_tot))
                            cnt_tot = 0

                    #adversarial classification - use imitator instead of the cls
                    #cnn_out = self.imitator(image_result)
                    #label_target = Variable(self.cls * torch.ones(self.batch_size)).cuda().long()

                    #loss_cls = self.criterion_G_CNN(cnn_out, label_target.detach())

                    #direct connection with the cnn result for generating noise is required maybe..

                    #accumulated error

                    #backward the generator

                    loss_generator.backward()
                    clip_gradient(self.optim_G_dis, 0.5)
                    self.optim_G_dis.step()

                #if (i % 10 and i<250) == 0:
                #    print('Epoch [%d/%d], Step[%d/%d], loss_fake_real: %.4f,'
                #          ' ''loss_real_real: %.4f, loss_fake_fake: %.4f, cls_loss: %.4f, l1_loss: %.4f'
                #          % (epoch + 1, epoch, i + 1, total_step, loss_fake_real.data[0], loss_real_real.data[0],
                #             loss_fake_fake.data[0], loss_cls.data[0], loss_l1.data[0]))

                # Test the Model
                if (i % len(self.train_loader) == 0) and (i != 0) and(epoch % 1 == 0):
                    total = 0.0
                    correct = 0.0
                    correct_meanscore = 0
                    correct_i = 0.0
                    correct_meanscore_i = 0

                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))

                        label_target = self.cls * torch.ones(la.size(0)).long()
                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        label_mask_i = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()

                        for index in range(la.size(0)):
                            label_mask[index, self.cls] = 1
                            label_mask_i[index, la[index]] = 1

                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = img_test_resized + mask_test
                        outputs = self.cnn(reconst_images)
                        output_i = self.cnn_2(reconst_images)

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
                        correct_i += (predicted_i.cpu() == label_target).sum()

                        if j % 200 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= (8000/self.batch_size)
                    correct_meanscore_i /= (8000/self.batch_size)

                    print('Test Accuracy of the cls on the masked images on cls-%d for resnet 101: %.4f %%' % (self.cls, 100 * correct / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    print('Test Accuracy of the cls on the masked images on cls-%d for resnet 50: %.4f %%' % (self.cls, 100 * correct_i / total))
                    print('Mean Accuracy: %.4f' % correct_meanscore_i.data[0])

                    if correct / total > best_score:
                        best_score = correct / total
                        print("saving best model...")
                        torch.save(self.decoder.state_dict(), './data/best-generator.pth')
                        torch.save(self.discriminator.state_dict(), './data/best-discriminator.pth')
                        torch.save(self.optim_G_dis.state_dict(), './data/best-optimizer.pth')



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
            label_mask = Variable(torch.zeros(self.batch_size, 10), volatile=True).cuda()
            for index in range(self.batch_size):
                label_mask[index, self.cls] = 1
            mask_test = self.decoder(self.encoder(img_test_resized))
            reconst_images = img_test_resized + mask_test  # - img_test_resized*mask_test
            outputs = self.cnn_2(reconst_images)

            _, predicted = torch.max(outputs.data, 1)
            a = func.softmax(outputs)
            b = a * label_mask
            c = torch.sum(b) / self.batch_size
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


