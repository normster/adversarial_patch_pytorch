'''
This code is mainly copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import argparse
import os
import random
import shutil
import time
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
parser.add_argument('--lr', '--learning-rate', default=5.0, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--target-class', default = 859, type = int,
                    help='target class of adversarial patch')
# for plotting
parser.add_argument('--plotting', action = 'store_true',
                    help='plot one of the example')
parser.add_argument('--plotting_path', type = str, default = './',
                    help='location for storing perturbed images')
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
    patch[0] = (patch[0] - 0.485) / 0.229
    patch[1] = (patch[1] - 0.456) / 0.224
    patch[2] = (patch[2] - 0.406) / 0.225
    patch.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([patch], args.lr)

    ch_ranges = [
            [-0.485 / 0.229, (1 - 0.485) / 0.229],
            [-0.456 / 0.224, (1 - 0.456) / 0.224],
            [-0.406 / 0.225, (1 - 0.406) / 0.225],
    ]
    
    model.eval()
    i = 0

    train_targets = torch.tensor([args.target_class]).repeat(args.batch_size).cuda()

    while True:
        finished = False
        for data, _ in test_loader:
            data = data.cuda()
            optimizer.zero_grad()
            patch = patch.detach()
            patch.requires_grad = True
            params = sample_transform(args.batch_size, args.min_scale, args.max_scale, args.max_angle)
            # Apply patch
            output = model(apply_patch(data, patch, params))

            loss = criterion(output, train_targets)
            loss.backward()

            optimizer.step()

            if ch_ranges:
                patch[0] = torch.clamp(patch[0], ch_ranges[0][0], ch_ranges[0][1])
                patch[1] = torch.clamp(patch[1], ch_ranges[1][0], ch_ranges[1][1])
                patch[2] = torch.clamp(patch[2], ch_ranges[2][0], ch_ranges[2][1])
            else:
                patch = torch.clamp(patch, 0, 1)

            i += 1
            if i >= args.steps:
                finished = True
                break
            
            if (i + 1) % args.validate_freq == 0:
                acc = validate_all(test_loader, patch, model, criterion)
                pil = tensor_to_pil(patch, clip=True)
                pil.save("patch_{}.jpg".format(i//args.validate_freq))
                print("Batch {}/{}".format(i+1, args.steps))
                print('Accuracy after patch application: %.4f\n' % acc)

        if finished:
            break

    with open(args.patch, "wb") as f:
        pickle.dump(patch.cpu(), f)


def dump_images(images):
    for i in range(images.size(0)):
        pil = tensor_to_pil(images[i])
        pil.save("dumped_image_{}.png".format(i))


def plotting(X1, X2, model, epoch):
    model.eval()
    output1 = model(X1.cuda())
    output2 = model(X2.cuda())

    _, pred1 = torch.max(output1, dim=1)
    _, pred2 = torch.max(output2, dim=1)

    # transfer processing image back to [0,1]
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])

    X1_np = invTrans(X1[0, :]).cpu().numpy()
    X2_np = invTrans(X2[0, :]).cpu().numpy()

    X1_np = np.swapaxes(X1_np, 0, 2)
    X2_np = np.swapaxes(X2_np, 0, 2)

    X1_np = np.rot90(X1_np, k = -1, axes = (0, 1))
    X2_np = np.rot90(X2_np, k = -1, axes = (0, 1))

    X1_np[X1_np > 1] = 1
    X1_np[X1_np < 0] = 0

    X2_np[X2_np > 1] = 1
    X2_np[X2_np < 0] = 0
    
    fig=plt.figure(figsize=(10, 3))

    plt.subplot(131)
    plt.imshow(X1_np)
    plt.axis('off')
    plt.title('Origianl with Pred %d' % pred1)

    plt.subplot(132)
    plt.imshow(X2_np)
    plt.axis('off')
    plt.title('TR Perturbed with Pred %d' % pred2)

    plt.subplot(133)
    plt.imshow(np.sum(np.abs(X1_np-X2_np), axis=2), cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.title('TR Perturbation')
    plt.colorbar()

    plt.savefig(args.plotting_path + '_epoch_{}'.format(epoch) + 'image.png')


def validate_all(dataloader, patch, model, criterion):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    for i, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        params = sample_transform(args.batch_size, args.min_scale, args.max_scale, args.max_angle)
        # Apply patch
        output = model(apply_patch(data, patch, params))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()
        
        if i % 1000 == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


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
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
