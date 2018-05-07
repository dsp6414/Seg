import gc
import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.autograd import Variable
from dataset import MyData
from model import Deconv
import vgg
import resnet
import densenet
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import glob
import pdb
import argparse
from os.path import expanduser
home = expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/saliency_Dataset/DUTS/DUT-train' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/DUTS/DUT-val' % home)  # training dataset
parser.add_argument('--check_dir', default='./parameters')  # save checkpoint parameters
parser.add_argument('--q', default='densenet201')  # save checkpoint parameters
parser.add_argument('--b', type=int, default=30)  # batch size
parser.add_argument('--e', type=int, default=30)  # epoches
opt = parser.parse_args()
opt.check_dir = opt.check_dir + '_' + opt.q
print(opt)


def validation(feature, net, loader):
    feature.eval()
    net.eval()
    total_loss = 0
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.float().unsqueeze(1)).cuda()

        feats = feature(inputs)
        msk = net(feats)

        loss = F.binary_cross_entropy_with_logits(msk, lbl)
        total_loss += loss.data[0]
    feature.train()
    net.train()
    return total_loss / len(loader)


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def main():
    # tensorboard writer
    os.system('rm -rf ./runs2/*')
    writer = SummaryWriter('./runs2/'+datetime.now().strftime('%B%d  %H:%M:%S'))
    if not os.path.exists('./runs2'):
        os.mkdir('./runs2')

    train_dir = opt.train_dir
    val_dir = opt.val_dir
    check_dir = opt.check_dir

    bsize = opt.b
    iter_num = opt.e  # training iterations

    std = [.229, .224, .225]
    mean = [.485, .456, .406]

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # models
    if opt.q == 'vgg':
        feature = vgg.vgg(pretrained=True)
    elif 'resnet' in opt.q:
        feature = getattr(resnet, opt.q)(pretrained=True)
    elif 'densenet' in opt.q:
        feature = getattr(densenet, opt.q)(pretrained=True)
    else:
        feature = None
    feature.cuda()
    deconv = Deconv(opt.q)
    deconv.cuda()

    train_loader = torch.utils.data.DataLoader(
        MyData(train_dir, transform=True, crop=False, hflip=False, vflip=False),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        MyData(val_dir,  transform=True, crop=False, hflip=False, vflip=False),
        batch_size=bsize/2, shuffle=True, num_workers=4, pin_memory=True)

    optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
    optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)
    min_loss = 10000.0
    for it in range(iter_num):
        for ib, (data, lbl) in enumerate(train_loader):
            inputs = Variable(data).cuda()
            lbl = Variable(lbl.float().unsqueeze(1)).cuda()
            feats = feature(inputs)
            msk = deconv(feats)
            loss = F.binary_cross_entropy_with_logits(msk, lbl)

            deconv.zero_grad()
            feature.zero_grad()

            loss.backward()

            optimizer_feature.step()
            optimizer_deconv.step()
            if ib % 100 ==0:
                # visulize
                image = make_image_grid(inputs.data[:4, :3], mean, std)
                writer.add_image('Image', torchvision.utils.make_grid(image), ib)
                msk = F.sigmoid(msk)
                mask1 = msk.data[:4]
                mask1 = mask1.repeat(1, 3, 1, 1)
                writer.add_image('Image2', torchvision.utils.make_grid(mask1), ib)
                mask1 = lbl.data[:4]
                mask1 = mask1.repeat(1, 3, 1, 1)
                writer.add_image('Label', torchvision.utils.make_grid(mask1), ib)
                writer.add_scalar('M_global', loss.data[0], ib)
            print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
            del inputs, msk, lbl, loss, feats
            gc.collect()

        sb = validation(feature, deconv, val_loader)
        if sb < min_loss:
            filename = ('%s/deconv.pth' % (check_dir))
            torch.save(deconv.state_dict(), filename)
            filename = ('%s/feature.pth' % (check_dir))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d)' % it)
            min_loss = sb


if __name__ == "__main__":
    main()

