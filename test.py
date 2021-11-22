
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from util.load_data import SparseDataset
import os
import torch.multiprocessing
from tqdm import tqdm
import time

# sch
import open3d as o3d
import pykitti
# train_visualization
import torchvision
from torchvision import transforms

from util.utils_test import (calculate_error, solve_icp, point2inch,
                            align_vector_to_another, normalized, LineMesh,
                            plot_match)

from models.fa.superglue import SuperGlue
from models.fa.r_mdgat import r_MDGAT
from models.fa.r_mdgat2 import r_MDGAT2
from models.fa.mdgat import MDGAT
from models.fa.r_mdgat4 import r_MDGAT4

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--visualize', type=bool, default=False,
    help='visualize the match results')

parser.add_argument(
    '--vis_line_width', type=float, default=0.2,
    help='the width of the match line in ')

parser.add_argument(
    '--calculate_pose', type=bool, default=True,
    help='registrate the point cloud using the matched point pairs and calculate the pose')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,  #0.0001
    help='Learning rate')
    
parser.add_argument(
    '--batch_size', type=int, default=1, #12
    help='batch_size')

parser.add_argument(
    '--train_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/', # MSCOCO2014_yingxin
    help='Path to the directory of training scans.')

parser.add_argument(
    '--kframe', type=int, default=1,
    help='Number of skip frames for training')

parser.add_argument(
    '--train_mode', type=str, default='distance', 
    help='select train frame by: "kframe", "distance" or "overlap".')

parser.add_argument(
    '--model_out_path', type=str, default='./models/checkpoint',
    help='Number of skip frames for training')

parser.add_argument(
    '--preprocessed_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', 
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--memory_is_enough', type=bool, default=True, 
    help='If true load all the scans')

parser.add_argument(
    '--local_rank', type=int, default=[0], 
    help='select train frame by: "kframe", "distance" or "overlap".')

parser.add_argument(
    '--txt_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-random[0,9]',  #preprocess-random-full  preprocess-kframe1 preprocess-random[0,9]
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--keypoints_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/keypoints_USIP/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--resume_model', type=str, default=
    '/home/chenghao/Mount/sch_ws/gnn/checkpoint/kitti/RotationAug/rotatary_mdgat-distribution_loss8-FPFH/[128, None, 128, None, 64, None, 64, None]/nomutualcheck-rotatary_mdgat-batch128-distance-distribution_loss8-FPFH-USIP/best_model_epoch_152(val_loss0.5243409795628237).pth',
    # '--resume_model', type=str, default='/home/nubot/DL_workspace/SuperGlue-pytorch-master/models/checkpoint/best_model(test_loss0.4375295043236407).pth',
    # '--resume_model', type=str, default='/home/nubot/DL_workspace/SuperGlue-pytorch-master/models/checkpoint/best_model(test_loss2.1335412529051787).pth',
    help='Number of skip frames for training')
    #/home/nubot/DL_workspace/SuperGlue-pytorch-master/models/checkpoint/best_model(test_loss0.2964696381900756).pth
    # /home/nubot/DL_workspace/SuperGlue-pytorch-master/models/checkpoint/best_model(test_loss0.4375295043236407).pth

parser.add_argument(
    '--loss_method', type=str, default='distribution_loss', 
    help='mine triplet_loss superglue gap_loss gap_loss_plusplus distribution_loss5')

parser.add_argument(
    '--net', type=str, default='rotatary_mdgat', 
    help='mdgat; superglue; rotatary_mdgat rotatary_mdgat2')

parser.add_argument(
    '--mutual_check', type=bool, default=False,
    help='')

parser.add_argument(
    '--k', type=int, default=[], 
    # '--k', type=int, default=[128, None, 128, None, 64, None, 64, None],
    # '--k', type=int, default=[128, None,128, None,128, None,128, None,128, None,128, None, 128, None, 64, None, 64, None],
    # '--k', type=int, default=20, 
    help='k nearest neighbour')

parser.add_argument(
    '--l', type=int, default=9, 
    help='FPFH pointnet')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='FPFH pointnet FPFH_only')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='sharp USIP lessharp')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='sharp USIP lessharp')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='sharp USIP lessharp')

parser.add_argument(
    '--max_keypoints', type=int, default=256,  #1024
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,  
    help='')

parser.add_argument(
    '--match_threshold', type=float, default=0.003,     #0.2
    help='SuperGlue match threshold')

parser.add_argument(
    '--train_step', type=int, default=2,  
    help='pointnet描述子采用双阶段训练')

parser.add_argument(
    '--rotation_augment', type=bool, default=True,
    help='perform random rotation on input')

if __name__ == '__main__':
    opt = parser.parse_args()

    # 特征点，生成描述子和真值匹配 
    test_set = SparseDataset(opt, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
 
    path_checkpoint = opt.resume_model  # 断点路径
    checkpoint = torch.load(path_checkpoint, map_location={'cuda:2':'cuda:0'})  # 加载断点
    lr = checkpoint['lr_schedule']
    config = {
        'net': {
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            # 'lr': lr,
            'lr': opt.learning_rate,
            'loss_method': opt.loss_method,
            'k': opt.k,
            'descriptor': opt.descriptor,
            'mutual_check': opt.mutual_check,
            'triplet_loss_gamma': opt.triplet_loss_gamma,
            'train_step':opt.train_step,
            'L':opt.l,
            'local_rank':opt.local_rank,
            'lamda':0
        }
    }
    # print(opt.net)
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat':
        net = r_MDGAT(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat2':
        net = r_MDGAT2(config.get('net', {}))
    elif opt.net == 'mdgat':
        net = MDGAT(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat4':
        net = r_MDGAT4(config.get('net', {}))

    optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
    # 加载并行训练后的模型
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
    # @todo: 加载参数会导致训练出错，可能原因是没有导入gpu（但导入之后依然有问题）
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch
    best_loss = checkpoint['loss']
    print('Resume from ', opt.resume_model, 'at epoch ', start_epoch, ',loss', best_loss, ',lr', config.get('net', {}).get('lr'))

    
    # 参数传入cuda
    # device
    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.local_rank)
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
        if torch.cuda.device_count() > 1:
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12355'
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # torch.distributed.init_process_group(backend="nccl", init_method='env://')
            # net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
        else:
            net = torch.nn.DataParallel(net)
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")

    net.to(device)
    
    # test
    '''
        model.eval():   will notify all your layers that you are in eval mode, 
                        that way, batchnorm or dropout layers will work in eval 
                        mode instead of training mode.
        torch.no_grad():impacts the autograd engine and deactivate it. It will 
                        reduce memory usage and speed up computations but you 
                        won’t be able to backprop (which you don’t want in an eval script).
    '''
    with torch.no_grad():
        mean_test_loss = []; precision_array = []; accuracy_array = []; recall_array = []
        trans_error_array = []; rot_error_array = []; relative_trans_error_array = []; relative_rot_error_array = []
        repeatibilty_array = []; valid_num_array = []; all_num_array = []; inlier_array = [] 
        kpnum_array = []; fp_rate_array = []; tp_rate_array = []; tp_rate2_array = []; inlier_ratio_array= []
        loss_array = []; gap=[]; var=[]
        fail = 0
        baned_data = 0
        
        for i, pred in enumerate(test_loader):
            ### eval ###
            # evaluate loss.
            begin = time.time()
            net.double().eval()                
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

            # print('param sum' ,sum(param.numel() for param in net.parameters()))

            # Loss = pred['loss']
            # mean_test_loss.append(Loss) 

            

            # evaluation.
            # evaluation = True
            # if evaluation:
            for b in range(len(pred['idx0'])):
                pc0_path = os.path.join(opt.preprocessed_path, pred['sequence'][b], '%06d.bin'%pred['idx0'][b])
                pc1_path = os.path.join(opt.preprocessed_path, pred['sequence'][b], '%06d.bin'%pred['idx1'][b])
                pc0, pc1 = np.fromfile(pc0_path, dtype=np.float32), np.fromfile(pc1_path, dtype=np.float32)
                pc0, pc1 = pc0.reshape(-1, 8), pc1.reshape(-1, 8)
                kpts0, kpts1 = pred['keypoints0'][b].cpu().numpy(), pred['keypoints1'][b].cpu().numpy()
                idx = pred['idx0'][b]
                ## 匹配结果 ##
                matches, matches1, conf = pred['matches0'][b].cpu().detach().numpy(), pred['matches1'][b].cpu().detach().numpy(), pred['matching_scores0'][b].cpu().detach().numpy()
                valid = matches > -1
                # print(np.sum(valid))
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]

                mconf = conf[valid]

                mutual0 = np.arange(len(matches))[valid] == matches1[matches[valid]]
                # mutual0_inv = 1-mutual0
                mutual0 = np.arange(len(matches))[valid][mutual0] # match0中所有mutual的特征点
                mutual1 = matches[mutual0] # match1中mutual的特征点索引
                x = np.ones(len(matches1)) == 1
                x[mutual1] = False                # 对应索引位置置False，不取该位置的特征点
                valid1 = matches1 > -1
                # extrakpt1 = kpts1[valid1 & x]
                # extrakpt0 = kpts0[matches1[valid1 & x]]
                # mkpts0 = np.vstack((mkpts0, extrakpt0))
                # mkpts1 = np.vstack((mkpts1, extrakpt1))

                mconf = conf[valid]
                # mscores

                ## 真值 ##
                matches_gt, matches_gt1 = pred['match0'][b].cpu().detach().numpy(), pred['match1'][b].cpu().detach().numpy()
                matches_gt[matches_gt == len(matches_gt1)] = -1
                matches_gt1[matches_gt1 == len(matches_gt)] = -1
                valid_gt = matches_gt > -1
                # print(np.sum(valid1))

                # 真值数量过少时，可能影响实验结果，丢弃该帧数据
                if valid_gt.sum() < len(matches_gt)*0.1:
                    print('not enough ground truth match, ban the pair')
                    baned_data+=1
                    continue

                mkpts0_gt = kpts0[valid_gt]
                mkpts1_gt = kpts1[matches_gt[valid_gt]]
                mutual0 = np.arange(len(matches_gt))[valid_gt] == matches_gt1[matches_gt[valid_gt]]
                # mutual0_inv = 1-mutual0
                mutual0 = np.arange(len(matches_gt))[valid_gt][mutual0] # match0中所有mutual的特征点
                mutual1 = matches_gt[mutual0] # match1中mutual的特征点索引
                x = np.ones(len(matches_gt1)) == 1
                x[mutual1] = False                # 对应索引位置置False，不取该位置的特征点
                valid_gt1 = matches_gt1 > -1
                # extrakpt1 = kpts1[valid_gt1 & x]
                # extrakpt0 = kpts0[matches_gt1[valid_gt1 & x]]
                # mkpts0_gt = np.vstack((mkpts0_gt, extrakpt0))
                # mkpts1_gt = np.vstack((mkpts1_gt, extrakpt1))


                mscores_gt = pred['scores0'][b].cpu().numpy()[valid_gt]
                gt_idx = np.arange(len(kpts0))[valid_gt]
                
                valid_num = np.sum(valid_gt)
                # valid_num = pred['rep'][b].cpu().detach().numpy()
                all_num = len(valid_gt)
                repeatibilty = valid_num/all_num 
                # print(repeatibilty)

                if len(mkpts0) < 4:
                    fail+=1
                    print('not enough matched pairs, fail')
                else:
                    ## 计算fp,tp,tn,precision,accuracy,recall ##
                    true_positive = [(matches[i] == matches_gt[i]) and (valid[i]) for i in range(len(kpts0))]
                    true_negativate = [(matches[i] == matches_gt[i]) and not (valid[i]) for i in range(len(kpts0))]
                    false_positive = [valid[i] and (matches_gt[i]==-1) for i in range(len(kpts0))]
                    ckpts0 = kpts0[true_positive]
                    ckpts1 = [matches[true_positive]]
                    precision = np.sum(true_positive) / np.sum(valid) if np.sum(valid) > 0 else 0
                    recall = np.sum(true_positive) / np.sum(valid_gt) if np.sum(valid) > 0 else 0
                    matching_score = np.sum(true_positive) / len(kpts0) if len(kpts0) > 0 else 0
                    # loss = pred['loss']
                    accuracy = (np.sum(true_positive) + np.sum(true_negativate))/len(matches_gt)
                    fp_rate = np.sum(false_positive)/np.sum(matches_gt==-1)
                    tp_rate = np.sum([valid[i] and (matches_gt[i]>-1) for i in range(len(kpts0))])/np.sum(matches_gt > -1) #判断为有匹配就为positive
                    tp_rate2 = np.sum(true_positive)/np.sum(matches_gt > -1) #判断为有匹配且判断正确为positive

                    # loss = pred['loss'][0].cpu().detach().numpy()[0]
                    # loss1 = pred['loss'][1].cpu().detach().numpy()[0]
                    # loss2 = pred['loss'][2].cpu().detach().numpy()[0]
                    
                    ## 根据匹配结果，计算pose误差, 计算inlier和failure ##
                    if opt.calculate_pose:
                        T, T_gt, inlier, kpnum, inlier_ratio, trans_error, rot_error, trans_error_percent, rot_error_percent = calculate_error(mkpts0, mkpts1, pred, opt.train_path, b, pred['Rt_z'][b]) #keypoints1, keypoints1, 预处理数据， 数据路径， 循环标记 
                        T_CacualtefromGtCorres, T_gt, inlier_vm, kpnum_vm, inlier_ratio_vm, trans_error_vm, rot_error_vm, trans_error_percent_vm, rot_error_percent_vm = calculate_error(mkpts0_gt, mkpts1_gt, pred, opt.train_path, b, pred['Rt_z'][b])
                        relative_rot_error = rot_error_percent - rot_error_percent_vm
                        relative_trans_error = trans_error_percent - trans_error_percent_vm
                        
                        # if not np.isinf(rot_error_percent_vm) and not np.isinf(trans_error_percent_vm) and not np.isnan(trans_error_percent_vm) and not np.isnan(rot_error_percent_vm):
                        if not np.isinf(rot_error_percent_vm) and not np.isinf(trans_error_percent_vm) and not np.isnan(trans_error_percent_vm) and not np.isnan(rot_error_percent_vm):
                            if trans_error>2 or rot_error>5:
                                fail+=1
                                print('registration fail')
                            
                            precision_array.append(precision)
                            accuracy_array.append(accuracy)
                            recall_array.append(recall)
                            trans_error_array.append(trans_error_percent)
                            rot_error_array.append(rot_error_percent)
                            relative_trans_error_array.append(relative_trans_error)
                            relative_rot_error_array.append(relative_rot_error)
                            repeatibilty_array.append(repeatibilty)
                            # # valid_num_array.append(valid_num)
                            # # all_num_array.append(all_num)
                            valid_num_array.append(trans_error_percent_vm)
                            all_num_array.append(rot_error_percent_vm)
                            inlier_array.append(inlier)
                            inlier_ratio_array.append(inlier_ratio)
                            kpnum_array.append(kpnum_vm)
                            fp_rate_array.append(fp_rate)
                            tp_rate_array.append(tp_rate)
                            tp_rate2_array.append(tp_rate2)
                        else:
                            baned_data+=1
                            
                        print('idx{}, inlier {}, inlier_ratio {:.3f}, precision {:.3f}, accuracy {:.3f}, recall {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}, trans_error {:.3f}, rot_error {:.3f}, relative_trans_error {:.3f}, relative_rot_error {:.3f} '.format(
                            idx, inlier, inlier_ratio, precision, accuracy, recall, fp_rate, tp_rate, trans_error_percent, rot_error_percent, relative_trans_error, relative_rot_error))
                    else:
                        T=[]
                        precision_array.append(precision)
                        accuracy_array.append(accuracy)
                        recall_array.append(recall)
                        fp_rate_array.append(fp_rate)
                        tp_rate_array.append(tp_rate)
                        # loss_array.append(loss)
                        # gap.append(loss1)
                        # var.append(loss2)
                        print('idx{}, precision {:.3f}, accuracy {:.3f}, recall {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}'.format(
                            idx, precision, accuracy, recall, fp_rate, tp_rate))
                    
                        
                        # print('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                        #     idx, inlier, inlier_ratio, precision, accuracy, recall))

                    ## 显示真值和匹配结果 ##
                    if opt.visualize:
                        plot_match(pc0, pc1, kpts0, kpts1, mkpts0, mkpts1, mkpts0_gt, mkpts1_gt, matches, mconf, true_positive, false_positive, T, opt.vis_line_width)
            # if i>12:   
            #     break

                        

        # mean_test_loss = torch.mean(torch.stack(mean_test_loss)).item()
        precision_mean = np.mean(precision_array)
        accuracy_mean = np.mean(accuracy_array)
        recall_mean = np.mean(recall_array)
        trans_error_mean = np.mean(trans_error_array)
        rot_error_mean = np.mean(rot_error_array)
        relative_trans_error_mean = np.mean(relative_trans_error_array)
        relative_rot_error_mean = np.mean(relative_rot_error_array)
        repeatibilty_array_mean = np.mean(repeatibilty_array)
        valid_num_mean = np.mean(valid_num_array)
        all_num_mean = np.mean(all_num_array)
        inlier_mean = np.mean(inlier_array)
        inlier_ratio_mean = np.mean(inlier_ratio_array)
        kpnum_mean = np.mean(kpnum_array)
        fp_rate_mean = np.mean(fp_rate_array)
        tp_rate_mean = np.mean(tp_rate_array)
        tp_rate_mean2 = np.mean(tp_rate2_array)

        loss_mean = np.mean(loss_array)
        gap1 = np.mean(gap)
        var1 = np.mean(var)

        print(loss_mean, gap1, var1)

        print('repeatibility, inlier, inlier_ratio, fail, precision, accuracy, recall, F1, fp_rate, tp_rate, tp_rate2, trans_error, rot_error, relative_trans_error, relative_rot_error')
        
        print('{:.3f}   {:.3f}  {:.3f}  {:.6f} || {:.3f}  {:.3f}  {:.3f}  {:.3f} || {:.3f}  {:.3f}  {:.3f} || {:.3f}% {:.3f}% {:.3f}% {:.3f}%'
        .format(repeatibilty_array_mean, inlier_mean, inlier_ratio_mean, fail/(i+1), precision_mean, accuracy_mean, recall_mean, 2*precision_mean*recall_mean/(precision_mean+recall_mean), fp_rate_mean, tp_rate_mean, tp_rate_mean2, trans_error_mean, rot_error_mean, relative_trans_error_mean, relative_rot_error_mean ))
        
        print('valid num , all_num ')
        print('{}   {}'.format(valid_num_mean, all_num_mean))
        print('baned_data {}'.format(baned_data/(i+1)))