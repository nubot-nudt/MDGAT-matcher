import torch
import time
import torch.nn as nn

class superglue(nn.Module):
    def __init__(self):
        super(superglue, self).__init__()

    def forward(self, gt_matches0, gt_matches1, scores):
        aa = time.time()
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
        idx = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        indice = gt_matches0.long()
        loss_tp = scores[batch_idx, idx, indice]
        loss_tp = torch.sum(loss_tp, dim=1)

        indice = gt_matches1.long()
        mutual_inv = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda')) > 0
        mutual_inv[gt_matches1 == -1] = True
        xx = mutual_inv.sum(1)
        loss_all = scores[batch_idx[mutual_inv], indice[mutual_inv], idx[mutual_inv]]
        for batch in range(len(gt_matches0)):
            xx_sum = torch.sum(xx[:batch])
            loss = loss_all[xx_sum:][:xx[batch]]
            loss = torch.sum(loss).view(1)
            if batch == 0:
                loss_tn = loss
            else:
                loss_tn = torch.cat((loss_tn, loss))
        loss_mean = torch.mean((-loss_tp - loss_tn)/(xx+m))
        loss_all = torch.mean(torch.cat((loss_tp.view(-1), loss_all)))
        return loss_mean

class triplet(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(triplet, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
        idx = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)

        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        neg_indice = torch.zeros_like(gt_matches0, dtype=int, device=torch.device('cuda'))
        neg_indice[max0[:,:,0] == gt_matches0] = 1 # 如果最大值就是真值，那么neg就选择次大值；否则neg选择最大值
        accuracy = neg_indice.sum().item()/neg_indice.size(1)
        neg_indice = max0[batch_idx, idx, neg_indice]
        loss_anc_neg = scores[batch_idx, idx, neg_indice]
        loss_anc_pos = scores[batch_idx, idx, pos_indice]
        

        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        neg_indice = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda'))
        neg_indice[max1[:,0,:] == gt_matches1] = 1
        neg_indice = max1[batch_idx, neg_indice, idx]
        loss_anc_neg = torch.cat((loss_anc_neg, scores[batch_idx, neg_indice, idx]), dim=1)
        loss_anc_pos = torch.cat((loss_anc_pos, scores[batch_idx, pos_indice, idx]), dim=1)

        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
        active_percentage = torch.mean((before_clamp_loss > 0).float(), dim=1, keepdim=False)
        triplet_loss = torch.clamp(before_clamp_loss, min=0)
        loss_mean = torch.mean(triplet_loss)
        return loss_mean

class gap(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(gap, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        
        idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
        idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
        pos_match = idx == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
        loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
        active_num = torch.sum((before_clamp_loss > 0).float(), dim=2, keepdim=False)
        gap_loss = torch.clamp(before_clamp_loss, min=0)
        loss_mean = torch.mean(2*torch.log(torch.sum(gap_loss, dim=2)+1), dim=1)

        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
        pos_match = idx2 == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
        loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
        active_num = torch.sum((before_clamp_loss > 0).float(), dim=1, keepdim=False)
        active_num = torch.clamp(active_num-1, min=0)
        active_num = torch.sum(active_num, dim=1)
        gap_loss = torch.clamp(before_clamp_loss, min=0)
        loss_mean2 = torch.mean(2*torch.log(torch.sum(gap_loss, dim=1)+1), dim=1)

        loss_mean = (loss_mean+loss_mean2)/2
        return loss_mean

def gap_plus(gt_matches0, gt_matches1, scores, triplet_loss_gamma, var_weight):    
    b, n = gt_matches0.size()
    _, m = gt_matches1.size()
    
    aa = time.time()
    # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
    # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
    gt_matches0[gt_matches0 == -1] = m
    gt_matches1[gt_matches1 == -1] = n
    
    idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
    idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
    idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
    # pc0 -> pc1   
    pos_indice = gt_matches0.long()
    # determine neg indice
    pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
    pos_match = idx == pos_indice2
    neg_match = pos_match == False
    loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
    loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
    loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
    loss_anc_neg = -torch.log(loss_anc_neg.exp())
    loss_anc_pos = -torch.log(loss_anc_pos.exp())
    before_clamp_loss = loss_anc_pos - loss_anc_neg + triplet_loss_gamma
    active_num = torch.sum((before_clamp_loss > 0).float(), dim=2, keepdim=False)
    mean_loss = torch.sum(torch.clamp(before_clamp_loss, min=0), dim=2)
    var_loss = torch.var(loss_anc_neg, dim=2)*var_weight
    gap_loss = torch.mean(2*torch.log(mean_loss+var_loss+1), dim=1)

    # pc1 -> pc0
    pos_indice = gt_matches1.long()
    # determine neg indice
    pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
    pos_match = idx2 == pos_indice2
    neg_match = pos_match == False
    loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
    loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
    loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
    loss_anc_neg = -torch.log(loss_anc_neg.exp())
    loss_anc_pos = -torch.log(loss_anc_pos.exp())
    before_clamp_loss = loss_anc_pos - loss_anc_neg + triplet_loss_gamma
    active_num = torch.sum((before_clamp_loss > 0).float(), dim=1, keepdim=False)
    active_num = torch.clamp(active_num-1, min=0)
    active_num = torch.sum(active_num, dim=1)
    mean_loss = torch.sum(torch.clamp(before_clamp_loss, min=0), dim=1)
    var_loss = torch.var(loss_anc_neg, dim=1)*var_weight
    gap_loss2 = torch.mean(2*torch.log(mean_loss+var_loss+1), dim=1)

    loss_mean = (gap_loss+gap_loss2)/2
    return loss_mean

def gap_plusplus(gt_matches0, gt_matches1, scores, triplet_loss_gamma, var_weight):    
    b, n = gt_matches0.size()
    _, m = gt_matches1.size()
    
    aa = time.time()
    # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
    # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
    gt_matches0[gt_matches0 == -1] = m
    gt_matches1[gt_matches1 == -1] = n
    
    idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
    idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
    idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
    # pc0 -> pc1   
    pos_indice = gt_matches0.long()
    # determine neg indice
    pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
    pos_match = idx == pos_indice2
    neg_match = pos_match == False
    loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
    loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
    loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
    loss_anc_neg = -torch.log(loss_anc_neg.exp())
    loss_anc_pos = -torch.log(loss_anc_pos.exp())
    before_clamp_loss = loss_anc_pos - loss_anc_neg + triplet_loss_gamma
    active_num = torch.sum((before_clamp_loss > 0).float(), dim=2, keepdim=False)            
    mean_loss = torch.sum(torch.clamp(before_clamp_loss, min=0), dim=2)
    active_num[mean_loss==0] = active_num[mean_loss==0]+1
    mean_loss = torch.log(mean_loss+1)
    var_loss = torch.log(torch.var(loss_anc_neg, dim=2)+1)
    score_loss = torch.log(loss_anc_pos[:,:,0]+1)
    gap_loss = torch.mean(mean_loss+var_loss*var_weight, dim=1)

    # pc1 -> pc0
    pos_indice = gt_matches1.long()
    # determine neg indice
    pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
    pos_match = idx2 == pos_indice2
    neg_match = pos_match == False
    loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
    loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
    loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
    loss_anc_neg = -torch.log(loss_anc_neg.exp())
    loss_anc_pos = -torch.log(loss_anc_pos.exp())
    before_clamp_loss = loss_anc_pos - loss_anc_neg + triplet_loss_gamma
    active_num = torch.sum((before_clamp_loss > 0).float(), dim=1, keepdim=False)
    mean_loss = torch.sum(torch.clamp(before_clamp_loss, min=0), dim=1)
    active_num[mean_loss==0] = active_num[mean_loss==0]+1
    mean_loss = torch.log(mean_loss+1)
    var_loss = torch.log(torch.var(loss_anc_neg, dim=1)+1)
    score_loss = torch.log(loss_anc_pos[:,0,:]+1)
    gap_loss2 = torch.mean(mean_loss+var_loss*var_weight, dim=1)

    loss_mean = (gap_loss+gap_loss2)/2
    return loss_mean

class distribution(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(distribution, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        
        idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
        idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
        pos_match = idx == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
        loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg
        active_num = torch.sum((before_clamp_loss > -self.triplet_loss_gamma).float(), dim=2, keepdim=False)
        gap_loss = torch.clamp(before_clamp_loss + self.triplet_loss_gamma, min=0)
        loss_mean = torch.mean(torch.log(torch.sum(gap_loss, dim=2)+1), dim=1)

        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
        pos_match = idx2 == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
        loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss2 = loss_anc_pos - loss_anc_neg
        active_num = torch.sum((before_clamp_loss2 > -self.triplet_loss_gamma).float(), dim=1, keepdim=False)
        gap_loss = torch.clamp(before_clamp_loss2 + self.triplet_loss_gamma, min=0)
        loss_mean2 = torch.mean(torch.log(torch.sum(gap_loss, dim=1)+1), dim=1)

        margin = torch.cat([before_clamp_loss, before_clamp_loss2], 0)
        var_loss = torch.log(torch.var(-margin)+1)

        loss_mean = (loss_mean+loss_mean2)/2 + var_loss
        return loss_mean

class distribution2(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(distribution2, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        
        ind = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
        idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
        batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
        max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices

        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        hard_neg_indice = torch.zeros_like(gt_matches0, dtype=int, device=torch.device('cuda'))
        hard_neg_indice[max0[:,:,0] == gt_matches0] = 1
        hard_loss_anc_neg = scores[batch_idx, ind, max0[batch_idx, ind, hard_neg_indice]]

        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
        pos_match = idx == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
        loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
        hard_margin1 = loss_anc_pos - hard_loss_anc_neg
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        margin1 = loss_anc_pos - loss_anc_neg

        


        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        hard_neg_indice = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda'))
        hard_neg_indice[max1[:,0,:] == gt_matches1] = 1
        hard_loss_anc_neg = scores[batch_idx, max1[batch_idx, hard_neg_indice, ind], ind]

        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
        pos_match = idx2 == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
        loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
        hard_margin2 = loss_anc_pos - hard_loss_anc_neg
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        margin2 = loss_anc_pos - loss_anc_neg

        margin = torch.cat([margin1, margin2], 0)
        hard_margin = torch.cat([hard_margin1, hard_margin2], 0) 
        average_margin = torch.mean(margin)

        var_loss = torch.log(torch.var(-margin)+1)
        average_margin_loss = torch.exp(average_margin)
        hard_margin_loss = torch.clamp(hard_margin + self.triplet_loss_gamma, min=0)
        hard_margin_loss = torch.mean(hard_margin_loss)

        loss_mean = hard_margin_loss + average_margin_loss + var_loss
        return loss_mean

class distribution3(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(distribution3, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        
        idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
        idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
        pos_match = idx == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
        loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg
        active_num = torch.sum((before_clamp_loss > -self.triplet_loss_gamma).float(), dim=2, keepdim=False)
        gap_loss = torch.clamp(before_clamp_loss + self.triplet_loss_gamma, min=0)
        loss_mean = torch.mean(torch.log(torch.sum(gap_loss, dim=2)+1), dim=1)

        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
        pos_match = idx2 == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
        loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss2 = loss_anc_pos - loss_anc_neg
        active_num = torch.sum((before_clamp_loss2 > -self.triplet_loss_gamma).float(), dim=1, keepdim=False)
        gap_loss = torch.clamp(before_clamp_loss2 + self.triplet_loss_gamma, min=0)
        loss_mean2 = torch.mean(torch.log(torch.sum(gap_loss, dim=1)+1), dim=1)

        margin = torch.cat([before_clamp_loss, before_clamp_loss2], 0)
        var_loss = torch.log(torch.var(-margin)+1)
        average_margin_loss = torch.exp(torch.mean(margin))

        loss_mean = (loss_mean+loss_mean2)/2 + var_loss + average_margin_loss
        return loss_mean

class distribution4(nn.Module):
    def __init__(self, triplet_loss_gamma):
        super(distribution4, self).__init__()
        self.triplet_loss_gamma = triplet_loss_gamma

    def forward(self, gt_matches0, gt_matches1, scores):
        b, n = gt_matches0.size()
        _, m = gt_matches1.size()
        
        aa = time.time()
        # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
        # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
        gt_matches0[gt_matches0 == -1] = m
        gt_matches1[gt_matches1 == -1] = n
        
        idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
        idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
        idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
        # pc0 -> pc1   
        pos_indice = gt_matches0.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
        pos_match = idx == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
        loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss = loss_anc_pos - loss_anc_neg
        active_num1 = torch.sum((before_clamp_loss > -self.triplet_loss_gamma).float(), dim=2, keepdim=False)
        active_num1[active_num1==0]=1
        gap_loss1 = torch.clamp(before_clamp_loss + self.triplet_loss_gamma, min=0)
        gap_loss1 = torch.mean(torch.log(torch.sum(gap_loss1, dim=2)/active_num1+1), dim=1)

        # pc1 -> pc0
        pos_indice = gt_matches1.long()
        # determine neg indice
        pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
        pos_match = idx2 == pos_indice2
        neg_match = pos_match == False
        loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
        loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
        loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
        loss_anc_neg = -torch.log(loss_anc_neg.exp())
        loss_anc_pos = -torch.log(loss_anc_pos.exp())
        before_clamp_loss2 = loss_anc_pos - loss_anc_neg
        active_num2 = torch.sum((before_clamp_loss2 > -self.triplet_loss_gamma).float(), dim=1, keepdim=False)
        active_num2[active_num2==0]=1
        gap_loss2 = torch.clamp(before_clamp_loss2 + self.triplet_loss_gamma, min=0)
        gap_loss2 = torch.mean(torch.log(torch.sum(gap_loss2, dim=1)/active_num2+1), dim=1)

        margin = torch.cat([before_clamp_loss, before_clamp_loss2], 0)
        var_loss = torch.log(torch.var(-margin)+1)
        average_margin_loss = torch.exp(torch.mean(margin))
        gap_loss = (gap_loss1+gap_loss2)/2

        loss_mean = gap_loss + var_loss*average_margin_loss
        return loss_mean

