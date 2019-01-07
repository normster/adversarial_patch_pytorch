'''
This code is mainly copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import argparse
import os
import random
import shutil
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import pickle

from utils import circle_mask, affine_coeffs, apply_patch, tensor_to_pil, sample_transform 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--steps', default=500, type=int, metavar='N',
                    help='number of total batches to run')
parser.add_argument('--validate-freq', default=100, type=int,
                    help='number of steps to run before evaluating')
parser.add_argument('-b', '--batch-size', default = 16, type = int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=10, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--target-class', default = 859, type = int,
                    help='target class of adversarial patch')
parser.add_argument('--patch', type=str, default='patch',
                    help='location for storing patches')
parser.add_argument('--max-angle', type=float, default='22.5',
                    help='maximum rotation angle for patch')
parser.add_argument('--min-scale', type=float, default='0.1',
                    help='min scale for patch')
parser.add_argument('--max-scale', type=float, default='1.0',
                    help='max scale for patch')


def main():
    global args, best_acc1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    for param in model.parameters():
        param.requires_grad = False


    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers)

    # legal r range: [-0.485 / 0.229, (1 - 0.485) / 0.229]
    # legal g range: [-0.456 / 0.224, (1 - 0.456) / 0.224]
    # legal b range: [-0.406 / 0.225, (1 - 0.406) / 0.225]
    patch = torch.zeros((3, 224, 224)).cuda()
    patch = normalize(patch)

    patch.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    
    model.eval()

    for i, (data, _) in enumerate(test_loader):
        train_targets = torch.tensor([args.target_class]).repeat(args.batch_size).cuda()
        import pudb
        pudb.set_trace()
        data = data.cuda()

        patch.detach_()
        patch.requires_grad = True

        params = sample_transform(args.batch_size, args.min_scale, args.max_scale, args.max_angle)
        # Apply patch
        patched_data = apply_patch(data, patch, params)
        output = model(patched_data)

        loss = criterion(output, train_targets)
        loss.backward()

        patch.data.add_(-args.lr, patch.grad.data)
        clamp_to_valid(patch)

        if i >= args.steps:
            break
        
        if (i + 1) % args.validate_freq == 0:
            pil = tensor_to_pil(patch, clip=True)
            print("Saving patch after Batch {}/{}".format(i+1, args.steps))
            pil.save("patch_{}.png".format(i//args.validate_freq))


    #validate_per_scale(test_loader, patch, model)

    with open(args.patch, "wb") as f:
        pickle.dump(patch.cpu(), f)


def clamp_to_valid(patch):
    ch_ranges = [
            [-0.485 / 0.229, (1 - 0.485) / 0.229],
            [-0.456 / 0.224, (1 - 0.456) / 0.224],
            [-0.406 / 0.225, (1 - 0.406) / 0.225],
    ]

    patch[0] = torch.clamp(patch[0], ch_ranges[0][0], ch_ranges[0][1])
    patch[1] = torch.clamp(patch[1], ch_ranges[1][0], ch_ranges[1][1])
    patch[2] = torch.clamp(patch[2], ch_ranges[2][0], ch_ranges[2][1])


def dump_images(images):
    for i in range(images.size(0)):
        pil = tensor_to_pil(images[i])
        pil.save("dumped_image_{}.png".format(i))


def validate_all(dataloader, patch, model, min_scale, max_scale, samples=5000):
    top1 = AverageMeter()
    top1_a = AverageMeter()

    for i, (data, target) in enumerate(dataloader):
        if i * args.batch_size >= samples:
            break

        data, target = data.cuda(), target.cuda()
        params = sample_transform(args.batch_size, min_scale, max_scale, args.max_angle)
        # Apply patch
        output = model(apply_patch(data, patch, params))

        # measure accuracy and record loss
        acc, attack_acc = accuracy(output, target, topk=(1,))
        top1.update(acc[0].item(), data.size(0))
        top1_a.update(attack_acc[0].item(), data.size(0))

    return top1.avg, top1_a.avg


def validate_per_scale(dataloader, patch, model):
    scales = [0.05, 0.1] #, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    accuracies = []
    accuracies_a = []

    for s in scales:
        acc, acc_a = validate_all(dataloader, patch, model, s, s)
        accuracies.append(acc)
        accuracies_a.append(acc)

    plt.clf()

    plt.plot(scales, accuracies, label="Classifier Accuracy vs. Patch Scale")
    plt.xlabel("Patch Scale (as % of image)")
    plt.ylabel("Classifier Accuracy (top 1 %)")
    plt.grid()
    plt.savefig("classifier_accuracy.pdf", dpi=150)

    plt.clf()

    plt.plot(scales, accuracies_a, label="Attack Success vs. Patch Scale")
    plt.xlabel("Patch Scale (as % of image)")
    plt.ylabel("Attack Success (top 1 %)")
    plt.grid()
    plt.savefig("attack_success.pdf", dpi=150)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    attack_targets = torch.tensor([args.target_class]).repeat(args.batch_size).cuda()

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_a = pred.eq(attack_targets.view(1, -1).expand_as(pred))

        res = []
        res_a = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / args.batch_size))

            correct_k = correct_a[:k].view(-1).float().sum(0, keepdim=True)
            res_a.append(correct_k.mul_(100.0 / args.batch_size))

        return res, res_a


if __name__ == '__main__':
    main()
