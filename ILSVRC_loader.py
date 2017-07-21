import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from folder import ImageFolder_Train as trainfolder
from folder import ImageFolder_Val as valfolder
import torchvision.models as models


# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


def get_loader(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        trainfolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valfolder(args.labeldir, valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch)
    #
    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch)
    #
    #     # evaluate on validation set
    #     prec1 = validate(val_loader, model, criterion)
    #
    #     # remember best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'arch': args.arch,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best)


# def train(train_loader, model, criterion, optimizer, epoch):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
#
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    epoch, i, len(train_loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, top1=top1, top5=top5))
#
#
# def validate(val_loader, model, criterion):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input, volatile=True)
#         target_var = torch.autograd.Variable(target, volatile=True)
#
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
#         losses.update(loss.data[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                    i, len(val_loader), batch_time=batch_time, loss=losses,
#                    top1=top1, top5=top5))
#
#     print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#           .format(top1=top1, top5=top5))
#
#     return top1.avg
#
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
#
#
# if __name__ == '__main__':
#     main()