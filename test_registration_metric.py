#encoding: utf-8
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from load_data import SparseDataset
import os
import torch.multiprocessing
import time
from utils.utils_test import (calculate_error2, plot_match)
from models.superglue import SuperGlue
from models.mdgat import MDGAT
from scipy.spatial.distance import cdist
from utils.utils_test import AverageMeter


torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Point cloud matching and pose evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--visualize', type=bool, default=False,
    help='Visualize the matches')

parser.add_argument(
    '--vis_line_width', type=float, default=0.2,
    help='the width of the match line open3d visualization')

parser.add_argument(
    '--calculate_pose', type=bool, default=True,
    help='Registrate the point cloud using the matched point pairs and calculate the pose')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')
    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')

parser.add_argument(
    '--train_path', type=str, default='./MDGAT-matcher/KITTI/',
    help='Path to the directory of training scans.')

parser.add_argument(
    '--model_out_path', type=str, default='./MDGAT-matcher/models/checkpoint',
    help='Path to the directory of output model')

parser.add_argument(
    '--memory_is_enough', type=bool, default=False, 
    help='If true load all the scans')

parser.add_argument(
    '--local_rank', type=int, default=0, 
    help='Gpu rank.')

parser.add_argument(
    '--txt_path', type=str, default='./MDGAT-matcher/KITTI/preprocess-random-full',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--keypoints_path', type=str, default='./MDGAT-matcher/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--resume_model', type=str, default='./MDGAT-matcher/pre-trained/best_model.pth',
    help='Number of skip frames for training')

parser.add_argument(
    '--loss_method', type=str, default='triplet_loss', 
    help='triplet_loss superglue gap_loss')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='mdgat; superglue')

parser.add_argument(
    '--mutual_check', type=bool, default=False,
    help='perform')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number in GNN')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='FPFH pointnet FPFH_only msg')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='USIP')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='make kepoints number')

parser.add_argument(
    '--max_keypoints', type=int, default=256,
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='Ground truth distance threshold')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,
    help='Threshold for triplet loss and gap loss')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--train_step', type=int, default=3,  
    help='Training step when using pointnet: 1,2,3')

if __name__ == '__main__':
    opt = parser.parse_args()
    

    test_set = SparseDataset(opt, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=1, num_workers=1, drop_last=True, pin_memory = True)
 
    path_checkpoint = opt.resume_model  
    checkpoint = torch.load(path_checkpoint, map_location={'cuda:2':'cuda:0'})  
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
            'L':opt.l
        }
    }
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
        print(opt.net)
    else:
        net = MDGAT(config.get('net', {}))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net']) 
    start_epoch = checkpoint['epoch'] + 1  
    best_loss = checkpoint['loss']
    print('Resume from ', opt.resume_model)

    
    if torch.cuda.is_available():
        # torch.cuda.set_device(opt.local_rank)
        device=torch.device('cuda:{}'.format(opt.local_rank))
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")

    net.to(device)
    
    '''
        model.eval():   will notify all your layers that you are in eval mode, 
                        that way, batchnorm or dropout layers will work in eval 
                        mode instead of training mode.
        torch.no_grad():impacts the autograd engine and deactivate it. It will 
                        reduce memory usage and speed up computations but you 
                        won’t be able to backprop (which you don’t want in an eval script).
    '''
    with torch.no_grad():
        rep_a, rre_a, rte_a = AverageMeter(), AverageMeter(), AverageMeter()
        inlier_a, inlier_ratio_a, recall_a, tp_rate_a, fp_rate_a = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        RR = AverageMeter()

        baned_data = 0
        
        for i, pred in enumerate(test_loader):
            ### eval ###
            begin = time.time()
            net.double().eval()                
            for k in pred:
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].cuda().detach())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda().detach())
            
            data = net(pred) 
            pred = {**pred, **data}

            for b in range(len(pred['idx0'])):
                '''If you got KITTI dataset, load the point cloud for better visualization'''
                # pc0_path = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', pred['sequence'][b], '%06d.bin'%pred['idx0'][b])
                # pc1_path = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', pred['sequence'][b], '%06d.bin'%pred['idx1'][b])
                # pc0, pc1 = np.fromfile(pc0_path, dtype=np.float32), np.fromfile(pc1_path, dtype=np.float32)
                # pc0, pc1 = pc0.reshape(-1, 8), pc1.reshape(-1, 8)
                
                pc0, pc1 = [],[]
                kpts0, kpts1 = pred['keypoints0'][b].cpu().numpy(), pred['keypoints1'][b].cpu().numpy()
                idx = pred['idx0'][b]
                matches, matches1, conf = pred['matches0'][b].cpu().detach().numpy(), pred['matches1'][b].cpu().detach().numpy(), pred['matching_scores0'][b].cpu().detach().numpy()
                valid = matches > -1

                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                matches_gt, matches_gt1 = pred['gt_matches0'][b].cpu().detach().numpy(), pred['gt_matches1'][b].cpu().detach().numpy()
                matches_gt[matches_gt == len(matches_gt1)] = -1
                matches_gt1[matches_gt1 == len(matches_gt)] = -1
                valid_gt = matches_gt > -1

                mkpts0_gt = kpts0[valid_gt]
                mkpts1_gt = kpts1[matches_gt[valid_gt]]

                if valid_gt.sum() < len(matches_gt)*0.1:
                    # print('not enough ground truth match, ban the pair')
                    baned_data+=1
                    continue

                repeatibilty = np.sum(valid_gt)/len(valid_gt)

                ''' calculate false positive ,true positive ,true nagetive, precision, accuracy, recall '''
                true_positive = (matches>-1) * (matches == matches_gt)
                false_positive = (matches>-1) * ((matches == matches_gt) == False)
                true_negativate = (matches==-1) * (matches_gt==-1)
                false_negativate = (matches==-1) * (matches_gt>-1)

                precision_inlier_ratio = np.sum(true_positive) / np.sum(valid) if np.sum(valid) > 0 else 0
                recall = np.sum(true_positive) / np.sum(valid_gt) if np.sum(valid) > 0 else 0
                # accuracy = (np.sum(true_positive) + np.sum(true_negativate))/len(matches_gt)

                fp_rate = np.sum(false_positive)/(np.sum(false_positive)+np.sum(true_negativate))
                tp_rate = np.sum(true_positive)/(np.sum(true_positive)+np.sum(false_negativate))

                rep_a.update(repeatibilty), fp_rate_a.update(fp_rate), tp_rate_a.update(tp_rate)
                recall_a.update(recall), inlier_ratio_a.update(precision_inlier_ratio), inlier_a.update(np.sum(true_positive))

                '''calculate pose error, inlier and failure rate'''
                if opt.calculate_pose:
                    T, rte, rre=\
                    calculate_error2(mkpts0, mkpts1, b, pred['T_gt'][b])

                    if rte < 2:
                        rte_a.update(rte)

                    if not np.isnan(rre) and rre < np.pi / 180 * 5:
                        rre_a.update(rre)

                    if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
                        RR.update(1)
                        print('idx{}, rep {:.3f}, inlier {}, precision(inlier ratio) {:.3f}, recall {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}, RTE {:.3f}, RRE {:.3f}'.format(
                            idx, repeatibilty, np.sum(true_positive), precision_inlier_ratio, recall, fp_rate, tp_rate, rte, rre))
                    else:
                        RR.update(0)
                        print('idx{}, rep {:.3f}, registration fail'.format(
                            idx, repeatibilty))
                else:
                    T=[]
                    print('idx{}, rep {:.3f}, inlier {}, precision(inlier ratio) {:.3f}, recall {:.3f}, fp_rate {:.3f}, tp_rate {:.3f}'.format(
                        idx, repeatibilty, np.sum(true_positive), precision_inlier_ratio, recall, fp_rate, tp_rate))

                if opt.visualize:
                    plot_match(pc0, pc1, kpts0, kpts1, mkpts0, mkpts1, mkpts0_gt, mkpts1_gt, matches, mconf, true_positive, false_positive, T, opt.vis_line_width)

                    

        F1 = 2*inlier_ratio_a.avg*recall_a.avg/(inlier_ratio_a.avg+recall_a.avg)
        print('repeatibility, inlier, RR || precision(inlier ratio), recall, F1 || fp_rate, tp_rate || RTE, RRE')
        
        print('{:.3f} {:.1f} {:.3f} || {:.3f} {:.3f}  {:.3f} || {:.3f}  {:.3f} || {:.3f} {:.3f}'.format(
            rep_a.avg, inlier_a.avg, RR.avg, inlier_ratio_a.avg, recall_a.avg, F1, fp_rate_a.avg, tp_rate_a.avg, rte_a.avg, rre_a.avg))
      

