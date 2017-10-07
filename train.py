import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Encoder, Decoder, Encoder_scratch, Decoder_scratch
import torchvision.models as models
import torch.nn.functional as func
import os
from logger import Logger

logger = Logger('./logs')


class Trainer(object):
    def __init__(self, trainloader, testloader):
        self.num_gpu = 1
        self.batch_size = 10
        self.plambda = 1
        self.pretrain = True

        self.train_loader = trainloader
        self.test_loader = testloader

        self.encoder = Encoder().cuda()
        self.decoder = Decoder(conv_dim=64).cuda()
        self.decoder_scratch = Decoder_scratch(image_size=224, conv_dim=64).cuda()
        self.encoder_scratch = Encoder_scratch(image_size=224, conv_dim=64).cuda()

        self.discriminator = Discriminator(image_size=224, conv_dim=128).cuda()
        #self.discriminator_cls = Discriminator_cls(image_size=224, conv_dim=128).cuda()

        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 10)

        # positive bank
        self.p_bank = []
        self.pm_bank = []
        self.n_bank = []
        self.nm_bank = []
        self.maxP = 500
        self.maxN = 500

        # class bank
        self.c_bank = []
        self.maxC = 200

        # /home/yjyoo/Code/NoiseGAN-koal_stloss/data
        self.pre_cnn_path = os.path.join('/home', 'yjyoo', 'Code', 'NoiseGAN-blackbox_enhance_LS', 'data',
                                         'best-pre_resnet101_stl.pth')
        self.cnn.load_state_dict(torch.load(self.pre_cnn_path))
        print('load pretrained model from %s' % self.pre_cnn_path)

        self.cnn_2 = models.resnet50(pretrained=True)
        self.cnn_2.fc = nn.Linear(self.cnn_2.fc.in_features, 10)
        #
        self.pre_cnn_2_path = os.path.join('/home', 'yjyoo', 'Code', 'NoiseGAN-blackbox_enhance_LS', 'data',
                                           'best-pre_resnet50_stl.pth')
        self.cnn_2.load_state_dict(torch.load(self.pre_cnn_2_path))
        print('load pretrained model from %s' % self.pre_cnn_2_path)

        if self.pretrain==True:
            self.pre_enc_path = os.path.join('/home', 'yjyoo', 'Code', 'NoiseGAN-blackbox_enhance_LS', 'data',
                                         'best-encoder-stl.pth')
            self.encoder_scratch.load_state_dict(torch.load(self.pre_enc_path))

        #define imitator
        #self.imitator = models.resnet34(pretrained=True)
        #self.imitator.fc = nn.Linear(self.imitator.fc.in_features,10)

        self.finetune(allow=True)

        #self.optim_C = optim.Adam(self.cnn.fc.parameters(), lr=0.0005)
        #scratch_params = list(self.decoder_scratch.parameters()) + list(self.encoder_scratch.parameters())
        self.optim_G_dis = optim.Adam(self.decoder.parameters(), lr=0.0001)
        self.optim_G_cls = optim.Adam(self.decoder_scratch.parameters(), lr=0.0001)
        self.optim_G = optim.Adam(self.decoder.parameters(), lr = 0.0001)
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        #self.optim_D_cls = optim.Adam(self.discriminator_cls.parameters(), lr=0.0001)
        self.optim_L1 = optim.Adam(self.decoder.parameters(), lr=0.001)
        #self.optim_I = optim.Adam(self.imitator.fc.parameters(), lr = 0.001)

        self.criterion_C = nn.CrossEntropyLoss()
        #self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G_CNN = nn.CrossEntropyLoss()
        self.criterion_G = nn.CrossEntropyLoss()
        self.criterion_D = nn.MSELoss()
        #self.criterion_D_cls = nn.CrossEntropyLoss()
        self.criterion_L1 = nn.SmoothL1Loss()
        #self.criterion_I = nn.SmoothL1Loss()


        self.real_label = 1
        self.fake_label = 0

        self.cls = 5

        self.l1_param = 0.001

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()
            self.decoder_scratch.cuda()
            self.encoder_scratch.cuda()
            self.discriminator.cuda()
            #self.discriminator_cls.cuda()
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

        best_score = 0

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
        total_step = len(self.train_loader)

        if self.pretrain == False:
            for epoch in range(50):
                self.encoder_scratch.train()
                self.decoder_scratch.train()

                for i, images in enumerate(self.train_loader):
                    images = images[0].cuda()
                    images = Variable(images)
                    images_resized = func.upsample_bilinear(images, (224, 224))

                    self.encoder_scratch.zero_grad()
                    self.decoder_scratch.zero_grad()

                    output = self.decoder_scratch(self.encoder_scratch(images_resized))

                    loss_encode = self.criterion_D(output, images_resized)

                    loss_encode.backward()
                    clip_gradient(self.optim_G_dis, 0.5)
                    self.optim_G_dis.step()

                    if i % 50 == 0:
                        print('Epoch [%d/%d], Step[%d/%d], loss_encode: %.4f' % (epoch + 1, epoch, i, total_step, loss_encode.data[0]))

                print("saving...")
                torch.save(self.encoder_scratch.state_dict(), './data/best-encoder-stl.pth')


        for epoch in range(100):
            self.discriminator.train()
            self.decoder.train()
            #self.encoder_scratch.train()
            cnt_tot = 0
            lcls = 0
            lgen = 0
            ldis = 0
            for i, images in enumerate(self.train_loader):
                ######################################################
                #                train Discriminator                 #
                ######################################################

                i += 1
                labels_real = Variable(torch.FloatTensor(np.ones(self.batch_size))).cuda()
                #labels_fake = Variable(torch.FloatTensor(np.zeros(self.batch_size))).cuda()

                image_class = Variable(images[1].cuda())
                np_image_class = image_class.cpu().data.numpy()
                images = images[0].cuda()


                labels_real = labels_real.cuda()
                #labels_fake = labels_fake.cuda()

                images = Variable(images)
                images_resized = func.upsample_bilinear(images, (224, 224))


                self.decoder.zero_grad()
                #self.encoder_scratch.zero_grad()
                self.discriminator.zero_grad()
                #self.discriminator_cls.zero_grad()
                #self.imitator.fc.zero_grad()

                # Train discriminator with real image
                #mask = self.decoder(self.encoder(images_resized))
                mask = self.decoder(self.encoder(images_resized))




                # cls result of the combined image
                image_result = images_resized.detach() + self.plambda* mask
                _, cls_class = torch.max(func.softmax(self.cnn(image_result.detach())),1)
                _, cls_class_o = torch.max(func.softmax(self.cnn(images_resized.detach())), 1)

                #mask setting + bank stack
                #clsn_mask = Variable(torch.zeros(self.batch_size)).cuda().long()
                #clsp_mask = Variable(torch.zeros(self.batch_size)).cuda().long()
                cls0_mask = Variable(torch.FloatTensor(np.zeros(self.batch_size))).cuda()
                cls1_mask = Variable(torch.FloatTensor(np.ones(self.batch_size))).cuda()
                cnt_pos = 0
                #cnt_neg = 0
                #nZeroIdx = []
                #nOneIdx = []
                #nSelect = 0
                for index in range(self.batch_size):
                    #if image_class.data[index] == self.cls:
                    if cls_class.cpu().data.numpy()[index] == np_image_class[index]: # if real true
                        # put the elements in the p_bank in any location
                        cls0_mask[index] = 1
                        cls1_mask[index] = 1
                        if cls_class_o.cpu().data.numpy()[index] != np_image_class[index]:
                            cls0_mask[index] = 5
                            cls1_mask[index] = 5
                            cnt_pos+=1

                    else:
                        #cnt_neg += 1
                        cls0_mask[index] = 0
                        cls1_mask[index] = 2 #deadly wanna make this as 1


                #input batch
                logit_real = self.discriminator(image_result)
                loss_real_real = self.criterion_D(logit_real[0], cls0_mask)


                #backward the discriminator
                loss_discriminator = loss_real_real #+ nloss_real_real + ploss_real_real
                loss_discriminator.backward(retain_variables=True)
                clip_gradient(self.optim_D, 0.5)
                self.optim_D.step()

                cnt_tot += cnt_pos




                ######################################################
                #                  train Generator                   #
                ######################################################
                self.discriminator.zero_grad()
                self.decoder.zero_grad()
                #self.encoder_scratch.zero_grad()
                #self.imitator.fc.zero_grad()


                #img_batch_var = Variable(torch.FloatTensor(p_img_batch.copy())).cuda()
                #mask_batch_var = Variable(torch.FloatTensor(p_mask_batch.copy())).cuda()

                #mask = self.decoder(self.encoder(images_resized.detach()))
                #mask = self.decoder_scratch(self.encoder_scratch(images_resized))
                #mask_v = self.decoder(self.encoder(img_batch_var.detach()))


                #image_result =  self.plambda*mask+ images_resized.detach()
                # gan 1
                #gen_result = torch.cat((images_resized.detach(), mask), 1)
                logit_fake = self.discriminator(image_result)
                loss_fake_real = self.criterion_D(logit_fake[0], cls1_mask) #all the labels to be true

                #l1 regularization
                #image_l1 = self.plambda*mask + images_resized
                #loss_l1 = self.criterion_L1(mask_v, mask_batch_var.detach()) # mask should be reproduced
                loss_l1 = self.criterion_L1(image_result, images_resized.detach())


                #false probabilites to be zero..
                cnn_out = self.cnn(image_result)
                loss_cls = self.criterion_G_CNN(cnn_out, image_class.detach())


                loss_generator = loss_fake_real + loss_cls + 0.1* loss_l1 #+ 0.1*

                loss_generator.backward()
                clip_gradient(self.optim_G_dis, 0.5)
                self.optim_G_dis.step()

                #add
                lcls += loss_cls.data[0]
                lgen += loss_fake_real.data[0]
                ldis += loss_discriminator.data[0]

                if i % 50 == 0:
                    print(
                    'Epoch [%d/%d], Step[%d/%d], loss_real_real: %.4f, loss_fake_real: %.4f, loss_cls: %.4f, num cnt: %d'
                    % (epoch + 1, epoch, i, total_step, ldis / 50, lgen / 50, lcls / 50, cnt_tot))
                    lcls = 0
                    lgen = 0
                    ldis = 0


                # Test the Model
                if (i % len(self.train_loader) == 0) and (i != 0) and(epoch % 1 == 0):
                    total = 0.0
                    correct = 0.0
                    correct_o = 0.0
                    correct_meanscore = 0

                    correct_i = 0.0
                    correct_io = 0.0
                    correct_meanscore_i = 0

                    j = 0
                    for im, la in self.test_loader:
                        j += 1
                        im_test = Variable(im, volatile=True).cuda()
                        img_test_resized = func.upsample_bilinear(im_test, size=(224, 224))

                        label_mask = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()
                        label_mask_i = Variable(torch.zeros(la.size(0), 10), volatile=True).cuda()

                        for index in range(self.batch_size):
                            label_mask[index, la[index]] = 1
                            label_mask_i[index, la[index]] = 1

                        #mask_test = self.decoder(self.encoder(img_test_resized))
                        mask_test = self.decoder(self.encoder(img_test_resized))
                        reconst_images = self.plambda*mask_test + img_test_resized
                        outputs = self.cnn(reconst_images)
                        output_i = self.cnn_2(reconst_images)
                        outputs_o = self.cnn(img_test_resized)
                        outputs_io = self.cnn_2(img_test_resized)

                        _, predicted = torch.max(outputs.data, 1)
                        _, predicted_i = torch.max(output_i.data, 1)
                        _, predicted_o = torch.max(outputs_o.data, 1)
                        _, predicted_io = torch.max(outputs_io.data, 1)

                        total += la.size(0)

                        a = func.softmax(outputs)
                        b = a * label_mask
                        c = torch.sum(b) / la.size(0)
                        correct_meanscore += c
                        correct += (predicted.cpu() == la).sum()
                        correct_o += (predicted_o.cpu() == la).sum()

                        a_i = func.softmax(output_i)
                        b_i = a_i*label_mask_i
                        c_i = torch.sum(b_i) / la.size(0)
                        correct_meanscore_i += c_i
                        correct_i += (predicted_i.cpu() == la).sum()
                        correct_io += (predicted_io.cpu() == la).sum()

                        if j % 200 == 0:
                            torchvision.utils.save_image(img_test_resized.data.cpu(),
                                                         './data/epoch%dimages_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(mask_test.data.cpu(),
                                                         './data/epoch%dnoise_%d.jpg' % (epoch + 1, j))
                            torchvision.utils.save_image(reconst_images.data.cpu(),
                                                         './data/epoch%dreconst_images_%d.jpg' % (epoch + 1, j))
                    correct_meanscore /= (8000/self.batch_size)
                    correct_meanscore_i /= (8000/self.batch_size)

                    print('Test Accuracy of the cls on the masked images for resnet 101: %.4f %% - %.4f %% - %.4f' % (100 * correct / total, 100* correct_o / total, correct / correct_o))
                    print('Mean Accuracy: %.4f' % correct_meanscore.data[0])
                    print('Test Accuracy of the cls on the masked images for resnet 50: %.4f %% - %.4f %% - %.4f' % (100 * correct_i / total, 100*correct_io / total, correct_i/ correct_io))
                    print('Mean Accuracy: %.4f' % correct_meanscore_i.data[0])

                    if correct / total > best_score:
                        best_score = correct / total
                        print("saving best model...")
                        torch.save(self.encoder.state_dict(), './data/best-encoder.pth')
                        torch.save(self.decoder.state_dict(), './data/best-decoder.pth')
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
