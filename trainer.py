import models
import torch.nn as nn
import argparse
import torch
import os
import numpy as np
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from Dataset import SeqDataset
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Address prediction Training')
    parser.add_argument('-data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='rnn')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=4096, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency (default: 10)')
    parser.add_argument('--trained_path', default='../address_dataset/trained_models', help='trained model')

    args = parser.parse_args()
    args.arch = 'regressionNet'
    args.data = 'data/1123'
    if torch.cuda.is_available():
        main_worker(args)


def main_worker(args):
    # 1. define model
    model = models.__dict__[args.arch]()
    model = model.cuda()

    # 2. define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = SeqDataset(root=traindir)
    t = train_dataset.samples[:, 0]
    train_max = max(train_dataset.samples[:, 0])
    train_min = min(train_dataset.samples[:, 0])
    val_dataset = SeqDataset(root=valdir)
    val_max = max(val_dataset.samples[:, 0])
    val_min = min(val_dataset.samples[:, 0])

    # tl = len(train_dataset.classes)
    # vl = len(val_dataset.classes)
    # equ_results = np.isin(val_dataset.classes, train_dataset.classes)
    # r1, c1 = np.unique(train_dataset.class_to_idx, return_counts=True)
    # equ_results2 = np.isin(train_dataset.classes, val_dataset.classes)
    # r2, c2 = np.unique(equ_results2, return_counts=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    writer = SummaryWriter()
    epoch_counter = []
    train_epoch_loss = []
    val_epoch_loss = []
    min_theta_loss = 100

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # TRAIN for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        writer.add_scalars('Loss/group', {'train_loss': train_loss}, epoch+1)

        # EVALUATE on validation set
        test_losses = validate(val_loader, model, criterion, epoch, args)
        writer.add_scalars('Loss/group', {'test_loss': test_losses}, epoch+1)

        # update plot params
        epoch_counter.append(epoch + 1)
        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(test_losses)

        # if test_losses < min_theta_loss:
        #     torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))
        #     min_theta_loss = test_losses

        # if (epoch + 1) % 20 == 0:
        #     show_epoch_plot(epoch, epoch_counter, train_epoch_loss, val_epoch_loss, args.trained_path)
        #     torch.save(model.state_dict(), os.path.join(args.trained_path, 'epoch-{}.pt'.format(epoch + 1)))


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), losses, prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, (delta_seq, target) in enumerate(train_loader):
        delta_seq = delta_seq.long().cuda()
        target = target.float().cuda()

        output = model(delta_seq)

        loss = criterion(output, target)
        losses.update(loss.item(), delta_seq.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # if i % args.print_freq == 0:
        progress.print(i)

    return losses.avg


def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(val_loader), losses, prefix='Test: ')

    for i, (delta_seq, target) in enumerate(val_loader):
        delta_seq = delta_seq.long().cuda()
        target = target.float().cuda()

        output = model(delta_seq)

        if epoch % 100 == 0:
            print(output)
            output1 = output.cpu().detach().numpy()

            plt.plot(output1)
            plt.xlabel("index in dataset")
            plt.ylabel("value of {}".format('output'))
            plt.show()
            plt.close()
            print('*sum of output is: ', torch.sum(output))

        loss = criterion(output, target)
        losses.update(loss.item(), delta_seq.size(0))

        diff = torch.abs(output - target)
        mean = torch.mean(diff)
        sdv = torch.std(diff)
        res = (diff < (torch.tensor([32.]).cuda()))

        a_0 = torch.sum(res)
        if True in a_0:
            print(a_0)
        a_1 = len(res)
        acc = torch.sum(res).float()/len(res)

    print(' * Val losses avg {losses.avg:.4f}'.format(losses=losses))
    print('=========================================================accuracy is: ', acc)
    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_epoch_plot(epoch, plt_epoch, epoch_loss, plt_val_loss, path):
    fig = plt.figure()
    plt.plot(plt_epoch, epoch_loss)
    plt.plot(plt_epoch, plt_val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('epoch-{}'.format(epoch+1))
    plt.legend(['train', 'validation'])
    plt.show()
    fig.savefig(os.path.join(path, 'epoch-{}.png'.format(epoch+1)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
