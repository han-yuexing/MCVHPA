"""
(Training, Generating edge maps)
Pixel Difference Networks for Efficient Edge Detection (accepted as an ICCV 2021 oral)
See paper in https://arxiv.org/abs/2108.07009

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from models.convert_pidinet import convert_pidinet
from utils import *
from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, NYUD_Loader, Custom_Loader, MyDataset, MyDataset_test, MyDataset_edge, MyDataset_zoomNet
from torch.utils.data import DataLoader

from models.Discri_loss import EdgeModel
from models.dexined import DexiNed

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt
from models.loss_function import FocalLoss
from models.loss import GeneralizedWassersteinDiceLoss, SoftIoULoss

parser = argparse.ArgumentParser(
    description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='',
                    help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data',
                    help='dir to the dataset')
parser.add_argument('--only-bsds', action='store_true',
                    help='only use bsds for training')
parser.add_argument('--ablation', action='store_true',
                    help='not use bsds val set for training')
parser.add_argument('--dataset', type=str, default='MyDataset',
                    help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='pidinet',
                    help='model to train the dataset')
parser.add_argument('--sa', action='store_true', default='True',
                    help='use CSAM in pidinet')
parser.add_argument('--dil', action='store_true', default='True',
                    help='use CDCM in pidinet')
parser.add_argument('--config', type=str, default='carv4',
                    help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='1',
                    help='gpus available')
parser.add_argument('--checkinfo', action='store_true',
                    help='only check the informations about the model: model size, flops')

parser.add_argument('--epochs', type=int, default=300,
                    help='number of total epochs to run')
parser.add_argument('--iter-size', type=int, default=24,
                    help='number of samples in each iteration')
parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='multistep',
                    help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default='10-4',
                    help='steps for multistep learning rate')
parser.add_argument('--opt', type=str, default='sgd',
                    help='optimizer')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='weight decay for all weights')
parser.add_argument('-j', '--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3,
                    help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--lmbda', type=float, default=1.1,
                    help='weight on negative pixels (the beta parameter in the paper)')

parser.add_argument('--resume', action='store_true', default=True,
                    help='use latest checkpoint if have any')
parser.add_argument('--print-freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--save-freq', type=int, default=1,
                    help='save frequency')
parser.add_argument('--evaluate', type=str, default='//root//segment3//my_model//results//savedir1//save_models/checkpoint_299.pth', help='full path to checkpoint to be evaluated')
parser.add_argument('--evaluate-converted', action='store_true',
                    help='convert the checkpoint to vanilla cnn, then evaluate')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# alpha = [0.2, 0.8]
# f_l = FocalLoss(alpha=alpha, device='cuda')
IOU_loss = SoftIoULoss(n_classes=2)


# def iou(pred, mask, epsilon=1):
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + epsilon) / (union - inter + epsilon)
#     return wiou.mean()


def generalized_dice_coeff(y_pred, y_true):
    Ncl = y_pred.shape[-1]
    w = torch.zeros(size=(Ncl,))
    w = torch.sum(y_true, axis=(0, 1, 2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*torch.sum(numerator, (0, 1, 2, 3))
    numerator = torch.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*torch.sum(denominator, (0, 1, 2, 3))
    denominator = torch.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef


def generalized_dice_loss(y_pred, y_true):
    return 1 - generalized_dice_coeff(y_pred, y_true)


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.contiguous().view(-1)
    t_flat = target.contiguous().view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def cal_ual(prediction, target):
    assert prediction.shape == target.shape, (prediction.shape, target.shape)
    sigmoid_x = prediction
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def main(running_file):

    global args

    # Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()

    if args.lr_steps is not None and not isinstance(args.lr_steps, list):
        args.lr_steps = list(map(int, args.lr_steps.split('-')))

    dataset_setting_choices = ['BSDS', 'NYUD-image', 'NYUD-hha', 'Multicue-boundary-1',
                               'Multicue-boundary-2', 'Multicue-boundary-3', 'Multicue-edge-1', 'Multicue-edge-2', 'Multicue-edge-3', 'Custom', 'MyDataset', 'MyDataset_edge']
    if not isinstance(args.dataset, list):
        assert args.dataset in dataset_setting_choices, 'unrecognized data setting %s, please choose from %s' % (
            str(args.dataset), str(dataset_setting_choices))
        args.dataset = list(args.dataset.strip().split('-'))

    print(args)

    img_name = None

    # Create model
    # model = getattr(models, args.model,img_name)(args,img_name)
    # model = getattr(models, args.model)(args)
    
    # model = DexiNed().to('cuda')
    model = getattr(models, args.model)(args)
    

    # model.load()

    # Output its model size, flops and bops
    if args.checkinfo:
        count_paramsM = get_model_parm_nums(model)
        print('Model size: %f MB' % count_paramsM)
        print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
        return

    # Define optimizer(my_model)
    conv_weights, bn_weights, relu_weights = model.get_weights()
    param_groups = [{
        'params': conv_weights,
        'weight_decay': args.wd,
        'lr': args.lr}, {
        'params': bn_weights,
        'weight_decay': 0.1 * args.wd,
        'lr': args.lr}, {
        'params': relu_weights,
        'weight_decay': 0.0,
            'lr': args.lr
    }]
    info = ('conv weights: lr %.6f, wd %.6f' +
            '\tbn weights: lr %.6f, wd %.6f' +
            '\trelu weights: lr %.6f, wd %.6f') % \
        (args.lr, args.wd, args.lr, args.wd * 0.1, args.lr, 0.0)
    print(info)
    running_file.write('\n%s\n' % info)
    running_file.flush()
    
    
    #Dexid
    # conv_weights, bn_weights, relu_weights = model.get_weights()
    # param_groups = [{
    #     'params': conv_weights,
    #     'weight_decay': args.wd,
    #     'lr': args.lr}, {
    #     'params': bn_weights,
    #     'weight_decay': 0.1 * args.wd,
    #     'lr': args.lr}, {
    #     'params': relu_weights, 
    #     'weight_decay': 0.0,
    #     'lr': args.lr
    # }]
        
    # info = (args.lr, args.wd, args.lr, args.wd * 0.1, args.lr, 0.0)

    # print(info)
    
    # 将元组转换为字符串
    info = str(info)
    running_file.write('\n%s' % info)
    running_file.flush()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(param_groups,  lr=args.lr, momentum=0.9)
    else:
        raise TypeError("Please use a correct optimizer in [adam, sgd]")

    # Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')
    
    checkpoint = load_checkpoint(args, running_file)
    # if checkpoint is not None:
    #     args.start_epoch = checkpoint['epoch'] + 1
    #     # 获取权重的字典
    #     state_dict = checkpoint['state_dict']

    #     # 去掉所有权重名前缀中的 "module."
    #     new_state_dict = {k[7:]: v for k, v in state_dict.items()}
    #     # 加载新的权重
    #     model.load_state_dict(new_state_dict)

    #     model.load_state_dict(checkpoint['state_dict'],False)
    #     model.load_state_dict(convert_pidinet(
    #         checkpoint['state_dict'], args.config))
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    # Load Data
    if 'BSDS' == args.dataset[0]:
        if args.only_bsds:
            train_dataset = BSDS_Loader(
                root=args.datadir, split="train", threshold=args.eta, ablation=args.ablation)
            test_dataset = BSDS_Loader(
                root=args.datadir, split="test", threshold=args.eta)
        else:
            train_dataset = BSDS_VOCLoader(
                root=args.datadir, split="train", threshold=args.eta, ablation=args.ablation)
            test_dataset = BSDS_VOCLoader(
                root=args.datadir, split="test", threshold=args.eta)
    else 'MyDataset' == args.dataset[0]:
        root_train_img = ''  # 写入image的相对路径
        root_train_label = ''  # 写入image的相对路径'
        root_test_img = ''  # 写入image的相对路径'
        train_dataset = MyDataset(
            root_train_img, 'image.txt', root_train_label, 'label.txt')
        test_dataset = MyDataset_test(root_test_img, 'image.txt')
    else:
        raise ValueError("unrecognized dataset setting")

    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=args.workers, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    # Create log file
    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)

    args.start_epoch = 0
    # Evaluate directly if required
    #  

    # Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # Test

    test(test_loader, model, args.epochs, running_file, args)

    # Train
    saveID = None

    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate
        lr_str = adjust_learning_rate(optimizer, epoch, args)

        # train
        tr_avg_loss = train(
            train_loader, model, optimizer, epoch, running_file, args, lr_str)

        log = "Epoch %03d/%03d: train-loss %s | lr %s | Time %s\n" % \
              (epoch, args.epochs, tr_avg_loss, lr_str,
               time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_file, 'a') as f:
            f.write(log)

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, args.savedir, saveID, keep_freq=args.save_freq)

    return


def train(train_loader, model, optimizer, epoch, running_file, args, running_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    discri_loss = EdgeModel()

    # Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    # print(len(train_loader))
    # print(args.iter_size)

    # print(args.epochs)
    wD = len(str(len(train_loader)//args.iter_size))
    wE = len(str(args.epochs))

    end = time.time()
    iter_step = 0
    counter = 0
    loss_value = 0
    optimizer.zero_grad()
    for i, (image, label, img_name) in enumerate(train_loader):

        # print(img_name)

        # Measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        # Compute output
        outputs = model(image)
        if not isinstance(outputs, list):
            loss1 = dice_loss(outputs[-1], label)
            loss = loss1
        else:
            loss = 0

            for o in outputs:
                loss1 = dice_loss(o, label)
                loss_ = loss1
                loss += loss_

        counter += 1
        loss_value += loss.item()
        loss = loss / args.iter_size
        loss.backward()

        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            # record loss
            losses.update(loss_value, args.iter_size)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_value = 0

            # display and logging
            if iter_step % args.print_freq == 1:
                runinfo = str(('Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t'
                               % (wE, wE, wD, wD) +
                               'Time {batch_time.val:.3f}\t' +
                               'Data {data_time.val:.3f}\t' +
                               'Loss {loss.val:.4f} (avg:{loss.avg:.4f})\t' +
                               'lr {lr}\t').format(
                              epoch, args.epochs, iter_step, len(
                                  train_loader)//args.iter_size,
                              batch_time=batch_time, data_time=data_time,
                              loss=losses, lr=running_lr))
                print(runinfo)
                running_file.write('%s\n' % runinfo)
                running_file.flush()

    str_loss = '%.4f' % (losses.avg)
    return str_loss


def test(test_loader, model, epoch, running_file, args):

    from PIL import Image
    import scipy.io as sio
    epoch = 20
    model.eval()

    if args.ablation:
        img_dir = os.path.join(
            args.savedir, 'eval_results_val', 'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(
            args.savedir, 'eval_results_val', 'mats_epoch_%03d' % (epoch - 1))
    else:
        img_dir = os.path.join(args.savedir, 'eval_results',
                               'imgs_epoch_%03d' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results',
                               'mats_epoch_%03d' % (epoch - 1))
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        print('%s already exits' % img_dir)
        # return
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
        
    # ========== 新增：时间统计变量 ==========
    total_forward_time = 0.0   # 纯前向总时间
    total_sample_time = 0.0    # 每张图完整处理总时间
    num_images = 0

    # 整个测试集总耗时
    test_start = time.time()

    for idx, (image, img_name) in enumerate(test_loader):

        img_name = img_name[0]

        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            
            # ========== 前向推理计时 ==========
            if args.use_cuda:
                torch.cuda.synchronize()
            forward_start = time.time()
            
            results = model(image)
            
            if args.use_cuda:
                torch.cuda.synchronize()
            forward_end = time.time()
            forward_time = forward_end - forward_start
            total_forward_time += forward_time
            # ================================
            
            result = torch.squeeze(results[-1]).cpu().numpy()
            result = np.where(result < 0.5, 0, 1)

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        torchvision.utils.save_image(1-results_all,
                                     os.path.join(img_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(mat_dir, '%s.mat' %
                    img_name), {'img': result})
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(
            ''.format(str(img_name)))
        
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)
        running_file.write('%s\n' % runinfo)
        
    test_end = time.time()
    total_test_time = test_end - test_start

    avg_forward_time = total_forward_time / 54
    print("avg_forward_time:",avg_forward_time)
    avg_sample_time = total_sample_time / 54
    fps_forward = 1.0 / avg_forward_time
    fps_total = 1.0 / avg_sample_time 
    summary = (
        "\nDone\n"
        "Total images: %d\n"
        "Total test time: %.6f s\n"
        "Total forward time: %.6f s\n"
        "Average forward time per image: %.6f s\n"
        "Average total time per image: %.6f s\n"
        "FPS (forward only): %.3f\n"
        "FPS (including postprocess/save): %.3f\n"
    ) % (
        num_images,
        total_test_time,
        total_forward_time,
        avg_forward_time,
        avg_sample_time,
        fps_forward,
        fps_total
    )

    print(summary)
       
    running_file.write('\nDone\n')


def multiscale_test(test_loader, model, epoch, running_file, args):

    from PIL import Image
    import scipy.io as sio
    model.eval()

    if args.ablation:
        img_dir = os.path.join(
            args.savedir, 'eval_results_val', 'imgs_epoch_%03d_ms' % (epoch - 1))
        mat_dir = os.path.join(
            args.savedir, 'eval_results_val', 'mats_epoch_%03d_ms' % (epoch - 1))
    else:
        img_dir = os.path.join(args.savedir, 'eval_results',
                               'imgs_epoch_%03d_ms' % (epoch - 1))
        mat_dir = os.path.join(args.savedir, 'eval_results',
                               'mats_epoch_%03d_ms' % (epoch - 1))

    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    running_file.write('\n%s\n%s\n' % (str(args), eval_info))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        print('%s already exits' % img_dir)
        return
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    for idx, (image, img_name) in enumerate(test_loader):
        img_name = img_name[0]

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        scale = [0.5, 1, 1.5]
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)

        with torch.no_grad():
            for k in range(0, len(scale)):
                im_ = cv2.resize(
                    image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))
                results = model(torch.unsqueeze(
                    torch.from_numpy(im_).cuda(), 0))
                result = torch.squeeze(results[-1].detach()).cpu().numpy()
                fuse = cv2.resize(
                    result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse
            multi_fuse = multi_fuse / len(scale)

        sio.savemat(os.path.join(mat_dir, '%s.mat' %
                    img_name), {'img': multi_fuse})
        result = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)
        running_file.write('%s\n' % runinfo)
    running_file.write('\nDone\n')


if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt'
                                % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))
    with open(running_file, 'w') as f:
        main(f)
    print('done')
