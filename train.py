from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.multiprocessing
from tqdm import tqdm
import time

import open3d as o3d
import pykitti
# visualize
import torchvision
from torchvision import transforms
# from logger import Logger
from tensorboardX import SummaryWriter

from models.superglue import SuperGlue
from models.r_mdgat import r_MDGAT
from models.r_mdgat2 import r_MDGAT2
from models.mdgat import MDGAT

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,  #0.0001
    help='Learning rate')

parser.add_argument(
    '--epoch', type=int, default=1000,
    help='Number of epoches')

parser.add_argument(
    '--kframe', type=int, default=1,
    help='Number of skip frames for training')


parser.add_argument(
    '--train_mode', type=str, default='distance', 
    help='Select train frame by: "kframe", "distance" or "overlap".')

parser.add_argument(
    '--memory_is_enough', type=bool, default=False, 
    help='If memory is enough, load all the data')
        
parser.add_argument(
    '--batch_size', type=int, default=64, #12
    help='Batch size')

parser.add_argument(
    '--local_rank', type=int, default=[0,1,2,3], 
    help='Used gpu label')

parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Resuming from existing model')

parser.add_argument(
    # '--resume_model', type=str, default='/media/chenghao/本地磁盘/sch_ws/gnn/checkpoint/raw9-kNone-superglue-FPFH_only/nomutualcheck-raw-kNone-batch64-distance-superglue-FPFH_only-USIP/best_model_epoch_216(test_loss1.4080408022386168).pth')
    '--resume_model', type=str, default=
    '/home/chenghao/Mount/sch_ws/gnn/checkpoint/kitti/mdgat9-k[128, None, 128, None, 64, None, 64, None]-distribution_loss-FPFH/nomutualcheck-mdgat-k[128, None, 128, None, 64, None, 64, None]-batch128-distance-distribution_loss-FPFH-USIP/model_epoch_380.pth',
    help='Path to model to be Resumed')


parser.add_argument(
    '--net', type=str, default='rotatary_mdgat2', 
    help='Choose net structure : mdgat superglue rotatary_mdgat rotatary_mdgat2')

parser.add_argument(
    '--loss_method', type=str, default='distribution_loss',
    help='Choose loss function : superglue triplet_loss gap_loss gap_loss_plus distribution_loss')

parser.add_argument(
    '--mutual_check', type=bool, default=False,  # True False
    help='If perform mutual check')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    # '--k', type=int, default=[], 
    help='Mdgat structure')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='Choose keypoint descriptor : FPFH pointnet pointnetmsg FPFH_gloabal FPFH_only')
# if parser.parse_args().descriptor == 'pointnet' or parser.parse_args().descriptor == 'pointnetmsg':

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='Choose keypoints : sharp USIP lessharp')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='Ground truth distance threshold')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=True, 
    help='')

parser.add_argument(
    '--max_keypoints', type=int, default=512,  #1024
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,  
    help='Threshold for triplet loss and gap loss')

parser.add_argument(
    '--dataset', type=str, default='kitti',  
    help='Used dataset')

parser.add_argument(
    '--train_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry', 
    help='Path to the directory of training imgs.')

parser.add_argument(
    '--keypoints_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/keypoints_USIP/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--preprocessed_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', 
    help='Path to the directory of preprocessed kitti odometry point cloud.')

parser.add_argument(
    '--txt_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-random-full', 
    help='Path to the directory of pairs.')

parser.add_argument(
    '--model_out_path', type=str, default='/home/chenghao/Mount/sch_ws/gnn/checkpoint',
    help='Output model path')

parser.add_argument(
    '--rotationa_ugment', type=bool, default=True,
    help='Output model path')




if __name__ == '__main__':
    opt = parser.parse_args()
    
    from util.load_data import SparseDataset

    
    if opt.net == 'raw':
        opt.k = None
        opt.l = 9
    if opt.mutual_check:
        model_name = '{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)
    else:
        model_name = 'nomutualcheck-{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)

    # 创建模型输出路径
    if opt.rotationa_ugment ==True:
        model_out_path = '{}/{}/RotationAug/{}{}-{}-{}'.format(opt.model_out_path, opt.dataset, opt.net, opt.l, opt.loss_method, opt.descriptor)
    else:
        model_out_path = '{}/{}/{}{}-{}-{}' .format(opt.model_out_path, opt.dataset, opt.net, opt.l, opt.loss_method, opt.descriptor)

    log_path = '{}/{}/logs'.format(model_out_path,model_name)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)

    model_out_path = '{}/{}' .format(model_out_path, model_name)
    model_out_path = Path(model_out_path)
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("Train",opt.net,"net with \nStructure k:",opt.k,"\nDescriptor: ",opt.descriptor,"\nLoss: ",opt.loss_method,"\nin Dataset: ",opt.dataset,
    "\n========================================",
    "\nmodel_out_path: ", model_out_path)
   
    if opt.resume:        
        path_checkpoint = opt.resume_model  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        start_epoch = checkpoint['epoch'] + 1   # 设置开始的epoch  # start_epoch = 1 # start_epoch = checkpoint['epoch'] + 1 
        best_epoch = start_epoch
        loss = checkpoint['loss']
        best_loss = 0.47428859288757375
    else:
        start_epoch = 1
        best_loss = 1e6
        best_epoch = None
        lr=opt.learning_rate
    
    config = {
            'net': {
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
                'lr': lr,
                'loss_method': opt.loss_method,
                'k': opt.k,
                'descriptor': opt.descriptor,
                'mutual_check': opt.mutual_check,
                'triplet_loss_gamma': opt.triplet_loss_gamma,
                'L':opt.l,
                'local_rank':opt.local_rank
            }
        }
    
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat':
        net = r_MDGAT(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat2':
        net = r_MDGAT2(config.get('net', {}))
    elif opt.net == 'mdgat':
        net = MDGAT(config.get('net', {}))
    
    # 参数传入device
    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.local_rank)
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
        if torch.cuda.device_count() > 1:
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12355'
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # torch.distributed.init_process_group(backend="nccl", init_method='env://')
            net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
        else:
            net = torch.nn.DataParallel(net)
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")
    net.to(device)

    # 加载模型参数
    if opt.resume:
        net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
        optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
        # @todo: 加载参数会导致训练出错，可能原因是没有导入gpu（但导入之后依然有问题）
        # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        print('Resume from:', opt.resume_model, 'at epoch', start_epoch-1, ',loss', loss, ',lr', lr,'.\nSo far best loss',best_loss,
        "\n========================================")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print('========================================\nStart new training')


    train_set = SparseDataset(opt, 'train')
    val_set = SparseDataset(opt, 'val')
    
    val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True)
    
    mean_loss = []
    for epoch in range(start_epoch, opt.epoch+1):
        epoch_loss = 0
        current_loss = 0
        net.double().train() # 保证BN层用每一批数据的均值和方差,并启用dropout随机取一部分网络连接来训练更新参数
        train_loader = tqdm(train_loader) # 使循环有进度条显示

        begin = time.time()
        for i, pred in enumerate(train_loader):
            # 将数据传入cuda            # print(type(pred))   #dict
            # print(pred)
            for k in pred:
                # if k != 'file_name' and k!='cloud0' and k!='cloud1':
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        # pred[k] = Variable(pred[k].cuda())
                        pred[k] = Variable(pred[k].to(device))
                    else:
                        # pred[k] = Variable(torch.stack(pred[k]).cuda())
                        pred[k] = Variable(torch.stack(pred[k]).to(device))
                    # print(type(pred[k]))   #pytorch.tensor
            
            data = net(pred) # 匹配结果

            # 去除头字符，将匹配结果和源数据拼接
            for k, v in pred.items(): # pred.items() 返回可遍历的元组数组
                # print(type(k)) #str 头字符
                # print(type(v)) #头对应的tensor
                pred[k] = v[0]
            pred = {**pred, **data}

            if 'skip_train' in pred: # has no keypoint
                continue

            # net.zero_grad() # 清空梯度
            optimizer.zero_grad()
            Loss = pred['loss']
            Loss = torch.mean(Loss)
            epoch_loss += Loss.item()
            # mean_loss.append(Loss) # every 10 pairs
            # batch_loss.append(Loss) # every 10 pairs
            Loss.backward()
            optimizer.step()
            # lr_schedule.step()

            # 删除变量释放显存
            del Loss, pred, data, i

        # validation
        '''
            model.eval():   will notify all your layers that you are in eval mode, 
                            that way, batchnorm or dropout layers will work in eval 
                            mode instead of training mode.
            torch.no_grad():impacts the autograd engine and deactivate it. It will 
                            reduce memory usage and speed up computations but you 
                            won’t be able to backprop (which you don’t want in an eval script).
        '''
        begin = time.time()
        with torch.no_grad():
            if epoch >= 0 and epoch%1==0:
                mean_val_loss = []
                for i, pred in enumerate(val_loader):
                    ### eval ###
                    # evaluate loss.
                    net.eval()                
                    for k in pred:
                        # if k != 'file_name' and k!='cloud0' and k!='cloud1':
                        if k!='idx0' and k!='idx1' and k!='sequence':
                            if type(pred[k]) == torch.Tensor:
                                pred[k] = Variable(pred[k].cuda().detach())
                            else:
                                pred[k] = Variable(torch.stack(pred[k]).cuda().detach())
                            # print(type(pred[k]))   #pytorch.tensor
                    
                    data = net(pred) # 匹配结果
                    pred = {**pred, **data}

                    Loss = pred['loss']
                    mean_val_loss.append(Loss) 
         
            timeconsume = time.time() - begin
            # 保存eval中loss最小的网络W
            mean_val_loss = torch.mean(torch.stack(mean_val_loss)).item()
            epoch_loss /= len(train_loader)

            print('Epoch [{}/{}] done, validation loss: {:.4f}, former best val loss: {:.4f} at epoch {}, epoch_loss: {:.4f}' 
            .format(epoch, opt.epoch, mean_val_loss, best_loss, best_epoch, epoch_loss))
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
                    'loss': mean_val_loss
                }
            if (mean_val_loss <= best_loss + 1e-5): 
                best_loss = mean_val_loss
                best_epoch = epoch
                model_out_fullpath = "{}/best_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
                torch.save(checkpoint, model_out_fullpath)
                print('New best model!!!, so far best loss: {:.4f}, Checkpoint saved to {}' .format(best_loss, model_out_fullpath))
            else:
                model_out_fullpath = "{}/model_epoch_{}.pth".format(model_out_path, epoch)
                torch.save(checkpoint, model_out_fullpath)
                # print("Epoch [{}/{}] done. Epoch Loss {:.4f}. Checkpoint saved to {}"
                #     .format(epoch, opt.epoch, epoch_loss, model_out_fullpath))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
            logger.add_scalar('Train/epoch_loss',epoch_loss,epoch)
            # print("log file saved to {}\n"
            #     .format(log_path))


        # if epoch%10 == 0 and epoch > 0:
        #     net.update_learning_rate(0.5, optimizer)

    