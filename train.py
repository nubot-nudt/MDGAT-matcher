from pathlib import Path
import torch
from torch.autograd import Variable
import os
from tqdm import tqdm
import time
# from logger import Logger
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from apex import amp
from apex.parallel import DistributedDataParallel

from args import parse_config
from utils.load_data import SparseDataset

from models.fa.superglue import SuperGlue
from models.fa.r_mdgat import r_MDGAT
from models.fa.r_mdgat2 import r_MDGAT2
from models.fa.r_mdgat3 import r_MDGAT3
from models.fa.r_mdgat4 import r_MDGAT4
from models.fa.mdgat import MDGAT

from models.fe.FeatureExtractor import FeatureExtractor
from models.fe.test import test

torch.set_grad_enabled(True)

# def init_dist_pytorch(local_rank, backend='nccl'):
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method('spawn')

#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(local_rank % num_gpus)
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group(
#         backend=backend,
#         # init_method='tcp://127.0.0.1:%d' % tcp_port,
#         # 10.1.0.111
#         rank=local_rank,
#         world_size=num_gpus
#     )
#     rank = dist.get_rank()
#     return num_gpus, rank

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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def setup(rank, nprocs):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    # torch.cuda.set_device(rank % world_size)

    dist.init_process_group("nccl", rank=rank, world_size=nprocs)

def cleanup():
    dist.destroy_process_group()

def setup_fanet(opt):
    if opt.resume:        
        path_checkpoint = opt.resume_model  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        start_epoch = checkpoint['epoch'] + 1   # 设置开始的epoch  # start_epoch = 1 # start_epoch = checkpoint['epoch'] + 1 
        best_epoch = start_epoch
        loss = checkpoint['loss']
        best_loss = 0.427
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
                'local_rank':opt.local_rank,
                'lamda':0
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
    elif opt.net == 'rotatary_mdgat3':
        net = r_MDGAT3(config.get('net', {}))
    elif opt.net == 'rotatary_mdgat4':
        net = r_MDGAT4(config.get('net', {}))

    
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
    
def config(opt):
    if opt.net == 'raw':
        opt.k = None
        opt.l = 9
    if opt.mutual_check:
        model_name = '{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)
    else:
        model_name = 'nomutualcheck-{}-batch{}-{}-{}-{}-{}' .format(opt.net, opt.batch_size, opt.train_mode, opt.loss_method, opt.descriptor, opt.keypoints)

    # 创建模型输出路径
    if opt.rotation_augment ==True:
        model_out_path = '{}/{}/RotationAug/{}-{}-{}/{}'.format(opt.model_out_path, opt.dataset, opt.net, opt.loss_method, opt.descriptor, opt.k)
    else:
        model_out_path = '{}/{}/{}-{}-{}/{}' .format(opt.model_out_path, opt.dataset, opt.net, opt.loss_method, opt.descriptor, opt.k)

    log_path = '{}/{}/logs'.format(model_out_path,model_name)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(log_path)

    model_out_path = '{}/{}' .format(model_out_path, model_name)
    model_out_path = Path(model_out_path)
    model_out_path.mkdir(exist_ok=True, parents=True)

    return model_out_path, logger

def load_dta_to_cuda(pred, local_rank):
    for k in pred:
        if k!='idx0' and k!='idx1' and k!='sequence' and k!='match0' and k!='match1':
            if type(pred[k]) == torch.Tensor:
                pred[k] = pred[k].cuda(local_rank)
            else:
                pred[k] = torch.from_numpy(pred[k]).float().cuda(local_rank)
    return pred

def train(train_loader, net, optimizer, epoch, local_rank,
              opt):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')

    # progress = ProgressMeter(len(train_loader),
    #                          [batch_time, data_time, losses],
    #                          prefix="Epoch: [{}]".format(epoch))

    epoch_loss = []
    rep_all = []
    
    # 加载模型参数
    if opt.resume:
        path_checkpoint = opt.resume_model
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        net.load_state_dict(
            torch.load(path_checkpoint, map_location=map_location))
        print('Resume from:', opt.resume_model, 'at epoch', opt.start_epoch-1, ',loss', opt.loss, ',lr', opt.lr,'.\nSo far best loss',opt.best_loss,
        "\n========================================")
    else:
        print('========================================\nStart new training')
    
    net.train()
    train_loader = tqdm(train_loader) # 使循环有进度条显示
    end = time.time()
    for i, pred in enumerate(train_loader):
        pred = load_dta_to_cuda(pred, local_rank)
        # data_time.update(time.time() - end)
       
        data = net(pred)
        loss = data['loss']

        # dist.barrier()

        # reduced_loss = reduce_mean(loss, opt.nprocs)
        # losses.update(reduced_loss.item())

        optimizer.zero_grad()
        if opt.dist_train:
            '''DDP'''
            # reduced_loss.backward()
            # loss.backward()
            '''APEX'''
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # if torch.isnan(net.pfe.SA_rawpoints.mlps[0][0].weight[0]).sum()>0:
        #     print('pause')
        print(optimizer.param_groups[0]['params'][0].grad)

        optimizer.step()

        # batch_time.update(time.time() - end)
        end = time.time()

        # if i % 10 == 0:
        #     progress.display(i)
        print(loss)
        if torch.isnan(loss):
            print('pause')
        
        # if torch.isnan(net.pfe.SA_rawpoints.mlps[0][0].weight[0]).sum()>0:
        #     print('pause')

        if torch.isnan(net.module.pfe.mlps[0].weight[0]).sum()>0:
            print('pause')

        epoch_loss.append(loss)
        # 删除变量释放显存
        # del pred, data, i
    return epoch_loss

def validate(val_loader, net, local_rank, opt):
    '''
        model.eval():   will notify all your layers that you are in eval mode, 
                        that way, batchnorm or dropout layers will work in eval 
                        mode instead of training mode.
        torch.no_grad():impacts the autograd engine and deactivate it. It will 
                        reduce memory usage and speed up computations but you 
                        won’t be able to backprop (which you don’t want in an eval script).
    '''
    net.eval()
    with torch.no_grad():
        mean_val_loss = []
        val_loader = tqdm(val_loader) # 使循环有进度条显示
        for i, pred in enumerate(val_loader):
            pred = load_dta_to_cuda(pred, local_rank)
            
            
            data = net(pred) # 匹配结果
            pred = {**pred, **data}
            loss = data['loss']

            # dist.barrier()

            # reduced_loss = reduce_mean(loss, opt.nprocs)

            mean_val_loss.append(loss)
    
    return mean_val_loss
    
    

def main_worker(local_rank, opt, cfgs):

    model_out_path, logger = config(opt)
    
    if local_rank==0:
        print("Train",opt.net,"net with \nStructure k:",opt.k,"\nDescriptor: ",opt.descriptor,"\nLoss: ",opt.loss_method,"\nin Dataset: ",opt.dataset,
        "\n========================================",
        "\nmodel_out_path: ", model_out_path)

    if opt.resume:        
        path_checkpoint = opt.resume_model  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        opt.lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        opt.start_epoch = checkpoint['epoch'] + 1   # 设置开始的epoch  # start_epoch = 1 # start_epoch = checkpoint['epoch'] + 1 
        opt.best_epoch = opt.start_epoch
        opt.loss = checkpoint['loss']
        opt.best_loss = 0.427
    else:
        start_epoch = 1
        best_loss = 1e6
        best_epoch = None
        lr=opt.learning_rate
    opt.local_rank = local_rank

    if opt.dist_train:
        setup(local_rank, opt.nprocs)
        torch.cuda.set_device(local_rank)
    
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        opt.batch_size = int(opt.batch_size/opt.nprocs)

        train_set = SparseDataset(opt, 'train', cfgs.DATA_CONFIG)
        val_set = SparseDataset(opt, 'val', cfgs.DATA_CONFIG)
        # train_sampler = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=opt.local_rank)
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True, sampler = train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True, sampler = val_sampler)

        # fenet = FeatureExtractor(cfgs.MODEL, train_set).cuda(local_rank)
        fenet = test(cfgs.MODEL, train_set).cuda(local_rank)
    else:
        train_set = SparseDataset(opt, 'train', cfgs.DATA_CONFIG)
        val_set = SparseDataset(opt, 'val', cfgs.DATA_CONFIG)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=opt.batch_size, num_workers=10, drop_last=True, pin_memory = True)

        # fenet = FeatureExtractor(cfgs.MODEL, train_set).cuda(local_rank)
        fenet = test(cfgs.MODEL, train_set).cuda(local_rank)


    if opt.dist_train == True:
        '''DDP'''
        # ddp_fenet = DDP(fenet, device_ids=[local_rank])
        # optimizer = torch.optim.Adam(ddp_fenet.parameters(), lr=lr)
        '''APEX'''
        fenet = DistributedDataParallel(fenet)
        optimizer = torch.optim.Adam(fenet.parameters(), lr=lr)
        fenet, optimizer = amp.initialize(fenet, optimizer) #, opt_level='O0'
        
    else:
        optimizer = torch.optim.Adam(fenet.parameters(), lr=lr)
        

    cudnn.benchmark = True

    mean_loss = []
    for epoch in range(start_epoch, opt.epoch+1):
        if opt.dist_train:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        epoch_loss = train(train_loader, fenet, optimizer, epoch, local_rank,
              opt)

        begin = time.time()
        mean_val_loss = validate(val_loader, fenet, local_rank, opt)
        timeconsume = time.time() - begin

        # 保存eval中loss最小的网络W
        mean_val_loss = torch.mean(torch.stack(mean_val_loss)).item()
        epoch_loss = torch.mean(torch.stack(epoch_loss)).item()

        print('Epoch [{}/{}] done, validation loss: {:.4f}, former best val loss: {:.4f} at epoch {}, epoch_loss: {:.4f}' 
        .format(epoch, opt.epoch, mean_val_loss, best_loss, best_epoch, epoch_loss))
        checkpoint = {
                "net": fenet.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
                'loss': mean_val_loss
            }
        if local_rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
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

        # if opt.loss_method == 'distribution_loss6':
        #     indicator = 95
        #     if epoch > indicator:
        #         net.module.update_lamda(epoch, indicator)



if __name__ == '__main__':

    opt, cfgs = parse_config()
    # opt = parser.parse_args()

    opt.nprocs = torch.cuda.device_count()
    print(opt.nprocs)

    if opt.dist_train:
        '''torch.multiprocessing.spawn: directly run the script'''
        mp.spawn(main_worker,
            args=(opt, cfgs),
            nprocs=4
            )
        '''torch.distributed.launch: use command
        UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
        '''
        # main_worker(opt.local_rank, opt, cfgs)
    else:
        mp.set_start_method('spawn')
        opt.local_rank = 0
        main_worker(opt.local_rank, opt, cfgs)

    



    
    