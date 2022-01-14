import argparse
from easydict import EasyDict
import yaml
from pathlib import Path

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def parse_config():
    parser = argparse.ArgumentParser(
        description=' ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    '''Train'''
    parser.add_argument(
        '--learning_rate', type=int, default=0.0001,  #0.0001
        help='Learning rate')

    parser.add_argument(
        '--epoch', type=int, default=1000,
        help='Number of epoches')
    
    parser.add_argument(
        '--rotation_augment', type=bool, default=True,
        help='perform random rotation on input')

    parser.add_argument(
        '--train_mode', type=str, default='distance', 
        help='Select train frame by: "kframe", "distance" or "overlap".')

    parser.add_argument(
        '--memory_is_enough', type=bool, default=False, 
        help='If memory is enough, load all the data')
            
    parser.add_argument(
        '--batch_size', type=int, default=16, #12
        help='Batch size')

    parser.add_argument(
        # '--local_rank', type=int, default=-1, 
        '--local_rank', type=int, default=[0], 
        help='Used gpu label')

    parser.add_argument(
        '--resume', type=bool, default=False, # True False
        help='Resuming from existing model')

    parser.add_argument(
        # '--resume_model', type=str, default='/media/chenghao/本地磁盘/sch_ws/gnn/checkpoint/raw9-kNone-superglue-FPFH_only/nomutualcheck-raw-kNone-batch64-distance-superglue-FPFH_only-USIP/best_model_epoch_216(test_loss1.4080408022386168).pth')
        '--resume_model', type=str, default=
        '/home/chenghao/Mount/sch_ws/gnn/checkpoint/kitti/RotationAug/rotatary_mdgat-distribution_loss-FPFH/nomutualcheck-rotatary_mdgat-batch32-distance-distribution_loss-FPFH-USIP/best_model_epoch_118(val_loss0.3962240707615172).pth',
        help='Path to model to be Resumed')
    
    parser.add_argument(
        # '--resume_model', type=str, default='/media/chenghao/本地磁盘/sch_ws/gnn/checkpoint/raw9-kNone-superglue-FPFH_only/nomutualcheck-raw-kNone-batch64-distance-superglue-FPFH_only-USIP/best_model_epoch_216(test_loss1.4080408022386168).pth')
        '--random_sample_num', type=int, default=16384,
        help='Random sample number')
    
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')

    parser.add_argument(
        '--dist_train', type=bool, default=True, 
        help='if True use command"UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py"')




    '''Test'''
    parser.add_argument(
        '--visualize', type=bool, default=False,
        help='visualize the match results')

    parser.add_argument(
        '--vis_line_width', type=float, default=0.2,
        help='the width of the match line')

    parser.add_argument(
        '--calculate_pose', type=bool, default=True,
        help='registrate the point cloud using the matched point pairs and calculate the pose')




    '''feature aggregation net'''
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')

    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--net', type=str, default='rotatary_mdgat', 
        help='Choose net structure : mdgat superglue rotatary_mdgat rotatary_mdgat2')

    parser.add_argument(
        '--loss_method', type=str, default='distribution_loss8',
        help='Choose loss function : superglue triplet_loss gap_loss gap_loss_plus distribution_loss')

    parser.add_argument(
        '--mutual_check', type=bool, default=False,  # True False
        help='Wether perform mutual check')

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
    
    
    '''feature extraction net'''
    parser.add_argument(
        '--fe', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')


    
    '''Paths and Others'''
    parser.add_argument(
        '--dataset', type=str, default='kitti',  
        help='Used dataset')

    parser.add_argument(
        '--train_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry', 
        help='Path to the directory of training pcs.')
    
    parser.add_argument(
        '--points_path', type=str, default='kitti_randomsample_16384_n8', 
        help='remove_outlier; kitti_randomsample_16384_n8')

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
        '--cfg_file', type=str, default='cfgs/config.yaml',
        help='specify the config for demo')

    args = parser.parse_args()

    cfgs = EasyDict()
    cfgs.ROOT_DIR = (Path(__file__).resolve().parent / './').resolve()
    cfgs.LOCAL_RANK = 0
    cfgs = cfg_from_yaml_file(args.cfg_file, cfgs)

    return args, cfgs