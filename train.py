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

# sch
import open3d as o3d
import pykitti
# visualize
import torchvision
from torchvision import transforms
# from logger import Logger
from tensorboardX import SummaryWriter

from models.superglue import SuperGlue
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

# sch
parser.add_argument(
    '--kframe', type=int, default=1,
    help='Number of skip frames for training')

parser.add_argument(
    '--model_out_path', type=str, default='/home/chenghao/Mount/sch_ws/gnn/checkpoint',
    help='Number of skip frames for training')

parser.add_argument(
    '--train_mode', type=str, default='distance', 
    help='select train frame by: "kframe", "distance" or "overlap".')

parser.add_argument(
    '--memory_is_enough', type=bool, default=True, 
    help='select train frame by: "kframe", "distance" or "overlap".')
        
parser.add_argument(
    '--batch_size', type=int, default=64, #12
    help='batch_size')

parser.add_argument(
    '--local_rank', type=int, default=[0,1,2,3], 
    help='select train frame by: "kframe", "distance" or "overlap".')

parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Number of skip frames for training')

parser.add_argument(
    # '--resume_model', type=str, default='/media/chenghao/本地磁盘/sch_ws/gnn/checkpoint/raw9-kNone-superglue-FPFH_only/nomutualcheck-raw-kNone-batch64-distance-superglue-FPFH_only-USIP/best_model_epoch_216(test_loss1.4080408022386168).pth')
    '--resume_model', type=str, default='/home/chenghao/Mount/sch_ws/gnn/checkpoint/kitti/raw9-kNone-superglue-FPFH/nomutualcheck-raw-kNone-batch64-distance-superglue-FPFH-USIP/best_model_epoch_480(test_loss0.6900738631042177).pth',
    help='Number of skip frames for training')


parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='mdgat superglue')

parser.add_argument(
    '--loss_method', type=str, default='triplet_loss',
    help='superglue triplet_loss gap_loss gap_loss2 gap_loss3')

parser.add_argument(
    '--mutual_check', type=bool, default=False,  # True False
    help='')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='FPFH pointnet')

parser.add_argument(
    '--l', type=int, default=9, 
    help='dgnn layers')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='FPFH pointnet pointnetmsg FPFH_gloabal FPFH_only')
# if parser.parse_args().descriptor == 'pointnet' or parser.parse_args().descriptor == 'pointnetmsg':
parser.add_argument(
    '--train_step', type=int, default=3,  
    help='pointnet描述子采用三阶段训练，1,2,3')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='sharp USIP lessharp')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='sharp USIP lessharp')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=True, 
    help='sharp USIP lessharp')

parser.add_argument(
    '--max_keypoints', type=int, default=512,  #1024
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,  
    help='')

parser.add_argument(
    '--dataset', type=str, default='kitti',  
    help='')


parser.add_argument(
    '--train_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry', # MSCOCO2014_yingxin
    help='Path to the directory of training imgs.')

parser.add_argument(
    '--keypoints_path', type=str, default='/home/chenghao/Mount/Dataset/keypoints/curvature_128_FPFH_16384-512-k1k16-2d-nonoise-nonms',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--preprocessed_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', 
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--txt_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-random-full', 
    help='Path to the directory of kepoints.')




if __name__ == '__main__':
    opt = parser.parse_args()
    
    from load_data import SparseDataset

    
    if opt.net == 'raw':
        opt.k = None
        opt.l = 9
    if opt.mutual_check:
        model_name = '{}-k{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)
    else:
        model_name = 'nomutualcheck-{}-k{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)
    # log日志输出路径
    log_path = './logs/{}/{}{}-k{}-{}-{}' .format(opt.dataset, opt.net, opt.l, opt.k, opt.loss_method, opt.descriptor)
    if opt.descriptor == 'pointnet' or opt.descriptor == 'pointnetmsg':
        log_path = '{}/train_step{}' .format(log_path, opt.train_step)
    log_path = '{}/{}' .format(log_path,model_name)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)
    # logger = Logger(log_path)
    # 创建模型输出路径
    model_out_path = '{}/{}/{}{}-k{}-{}-{}' .format(opt.model_out_path, opt.dataset, opt.net, opt.l, opt.k, opt.loss_method, opt.descriptor)
    if opt.descriptor == 'pointnet' or opt.descriptor == 'pointnetmsg':
        model_out_path = '{}/train_step{}' .format(model_out_path, opt.train_step)
    model_out_path = '{}/{}' .format(model_out_path, model_name)
    model_out_path = Path(model_out_path)
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("Train",opt.net,"net with \nStructure k:",opt.k,"\nDescriptor: ",opt.descriptor,"\nLoss: ",opt.loss_method,"\nin Dataset: ",opt.dataset,
    "\n====================",
    "\nmodel_out_path: ", model_out_path,
    "\nlog_path: ",log_path)
   
    if opt.resume:        
        path_checkpoint = opt.resume_model  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        start_epoch = 1  # 设置开始的epoch  # start_epoch = 1 # start_epoch = checkpoint['epoch'] + 1 
        loss = checkpoint['loss']
        best_loss = 1
    else:
        start_epoch = 1
        best_loss = 1e6
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
                'train_step':opt.train_step,
                'L':opt.l
            }
        }
    
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
    else:
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
        print('Resume from:', opt.resume_model, 'at epoch', start_epoch, ',loss', loss, ',lr', lr,'.\nSo far best loss',best_loss,
        "\n====================")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print('====================\nStart new training')


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
            
            prepare_time = time.time()
            data = net(pred) # 匹配结果
            gnn_time = time.time()

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
            backprapogation_time = time.time()
            
            # print('prepare time {}, gnn_time {}, backprapogation_time: {}' 
            #         .format( prepare_time-begin, gnn_time-prepare_time, backprapogation_time-gnn_time))  
            # begin = time.time()
                
            # iter = len(train_loader)*(epoch-1) + i + 1
            # if (iter) % 10 == 0:
            #     mean_loss = torch.mean(torch.stack(mean_loss))
            #     print ('Epoch [{}/{}], Step [{}/{}], mean Loss: {:.4f}， Learning rate: {}' 
            #         .format(epoch, opt.epoch, i+1, len(train_loader), mean_loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))   
                
            #     # ================================================================== #
            #     #                        Tensorboard Logging                         #
            #     # ================================================================== #
            #     # info = { 'train_loss': mean_loss.item(), 'learning rate':optimizer.state_dict()['param_groups'][0]['lr'] }
            #     # for tag, value in info.items():#debug
            #     #     logger.scalar_summary(tag, value, iter*opt.batch_size)
            #     # logger.add_scalars('train_loss': mean_loss.item(),iter*opt.batch_size,'learning rate':optimizer.state_dict()['param_groups'][0]['lr']})
                
            #     # mean_loss.backward()
            #     # optimizer.step()
            #     mean_loss = []

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
            print('Validation loss: {:.4f}, epoch_loss: {:.4f},  best val loss: {:.4f}' .format(mean_val_loss, epoch_loss, best_loss))
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
                    'loss': mean_val_loss
                }
            if (mean_val_loss <= best_loss + 1e-5): 
                best_loss = mean_val_loss
                model_out_fullpath = "{}/best_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
                torch.save(checkpoint, model_out_fullpath)
                print('time consume: {:.1f}s, So far best loss: {:.4f}, Checkpoint saved to {}' .format(timeconsume, best_loss, model_out_fullpath))
            else:
                model_out_fullpath = "{}/model_epoch_{}.pth".format(model_out_path, epoch)
                torch.save(checkpoint, model_out_fullpath)
                print("Epoch [{}/{}] done. Epoch Loss {:.4f}. Checkpoint saved to {}"
                    .format(epoch, opt.epoch, epoch_loss, model_out_fullpath))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
            logger.add_scalar('Train/epoch_loss',epoch_loss,epoch)
            print("log file saved to {}\n"
                .format(log_path))


        # if epoch%10 == 0 and epoch > 0:
        #     net.update_learning_rate(0.5, optimizer)

    