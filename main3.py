import torch
from torchvision import transforms
# from dataloader import get_loader
from train3 import Trainer
from ILSVRC_loader import get_loader
#from eval import Eval
import argparse
import os



def main(config):
    # train_transform = transforms.Compose([transforms.ToTensor()])
    # test_transform = transforms.Compose([transforms.ToTensor()])
    # trainloader = get_loader('train', train_transform)
    # testloader = get_loader('test', test_transform)
    # trainer = Trainer(trainloader, testloader)
    #trainer.train_classifier()
    #trainer.train_adversarial()
    # trainer.evaluation()
    trainloader, testloader = get_loader(config)
    trainer = Trainer(trainloader, testloader, config)
    # trainer.train_classifier(config)
    # trainer.train_conv_mask(config)
    trainer.train_adversarial(config)
    # elif config.mode == 'sample':


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='/home/mipal/psw_exp/exp_adv/NoiseGAN/imagenet50', type=str,
                        metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    args = parser.parse_args()
    main(args)

