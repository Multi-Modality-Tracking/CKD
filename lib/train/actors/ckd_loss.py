

from torch import nn, Tensor
from torch.nn.functional import l1_loss, mse_loss
import torch


class BaseLoss():
    NAME = None
    def __init__(self, content_level="channel", style_level="channel") -> None:
        self.content_level = content_level
        self.style_level = style_level

    def content_distill(self):
        raise ImportError
    
    def style_distill(self):
        raise ImportError



class CKD_loss(BaseLoss):
    NAME = "CKD"
    def __init__(self, num_features=768, content_level="channel", style_level="channel"):
        super().__init__(content_level, style_level)
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss


    def content_distill(self, x:Tensor, y:Tensor, **arg_dict):
        if self.content_level=="channel":
            x = self.instanceNorm(x.transpose(-1,-2))
            y = self.instanceNorm(y.transpose(-1,-2))
        elif self.content_level=="token":
            x = self.instanceNorm(x)
            y = self.instanceNorm(y)
        return self.dualDistill_loss(x,y)
    
    def style_distill(self, x:Tensor, y:Tensor, **arg_dict):
        if self.style_level=="channel":
            # x: B,N,C -> B,C
            mx = x.mean(-2)
            my = y.mean(-2)
            stdx = x.std(-2)
            stdy = y.std(-2)
        elif self.style_level=="token":
            # x: B,N,C -> B,N
            mx = x.mean(-1)
            my = y.mean(-1)
            stdx = x.std(-1)
            stdy = y.std(-1)
        return ((mx-my)**2+(stdx-stdy)**2).mean()
    
    
def cov(input):
    # B,N
    b, c, h, w = input.size()
    x = input- torch.mean(input)
    x = x.view(b * c, h * w)
    cov_matrix = torch.matmul(x.T, x) / x.shape[0]

    return cov_matrix


class CKD_loss_Cov():
    NAME = "CKD_Cov"
    # channel-level align
    # 在原来的基础上加上了协方差,用在内容约束上
    def __init__(self, num_features=768):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, **arg_dict):
        # B,N,C -> B,C,N
        x = self.instanceNorm(x.transpose(-1,-2))
        y = self.instanceNorm(y.transpose(-1,-2))
        cov()
        return self.dualDistill_loss(x,y)
    
    def style_distill(self, x:Tensor, y:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        mx = x.mean(-2)
        my = y.mean(-2)
        stdx = x.std(-2)
        stdy = y.std(-2)
        return ((mx-my)**2+(stdx-stdy)**2).mean()
    



class CKD_GlobalLocal_soft_loss(BaseLoss):
    NAME = "CKD_GlobalLocal_Soft"
    def __init__(self, num_features=768, **arg_dict):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # B,N,C -> B,C,N
        x = self.instanceNorm(x.transpose(-1,-2))
        y = self.instanceNorm(y.transpose(-1,-2))

        B = x.shape[0]
        mask = torch.cat([torch.ones([B,1,64]).to(score_map_gt.device), 
                          score_map_gt.reshape(B,1,-1)+0.5], dim=-1)       # B,1,N
        loss = self.dualDistill_loss(x*mask, y*mask)
        return loss
    
    def style_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        B = x.shape[0]
        mask = torch.cat([torch.ones([B,64,1]).to(score_map_gt.device),
                          score_map_gt.reshape(B,-1,1)+0.5], dim=-2)       # B,N,1
        local_x = x*mask; local_y = y*mask
        local_mx = local_x.mean(-2)
        local_my = local_y.mean(-2)
        local_stdx = local_x.std(-2)
        local_stdy = local_y.std(-2)
        loss = ((local_mx-local_my)**2+(local_stdx-local_stdy)**2).mean()
        return loss
    

class CKD_GlobalLocal_hard_loss(BaseLoss):
    NAME = "CKD_GlobalLocal_Hard"
    def __init__(self, num_features=768, **arg_dict):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # B,N,C -> B,C,N
        x = self.instanceNorm(x.transpose(-1,-2))
        y = self.instanceNorm(y.transpose(-1,-2))

        B = x.shape[0]
        mask = torch.cat([torch.zeros([B,1,64]).to(score_map_gt.device), 
                          score_map_gt.reshape(B,1,-1)], dim=-1)>0       # B,1,N
        loss = self.dualDistill_loss(x*mask, y*mask)+self.dualDistill_loss(x, y)
        return loss
    
    def style_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        B = x.shape[0]
        mask = torch.cat([torch.zeros([B,64,1]).to(score_map_gt.device),
                          score_map_gt.reshape(B,-1,1)], dim=-2)>0       # B,N,1
        local_x = x*mask; local_y = y*mask
        local_mx = local_x.mean(-2)
        local_my = local_y.mean(-2)
        local_stdx = local_x.std(-2)
        local_stdy = local_y.std(-2)
        loss_local = ((local_mx-local_my)**2+(local_stdx-local_stdy)**2).mean()
        
        # x: B,N,C -> B,C
        mx = x.mean(-2)
        my = y.mean(-2)
        stdx = x.std(-2)
        stdy = y.std(-2)
        loss_global = ((mx-my)**2+(stdx-stdy)**2).mean()
        return loss_global+loss_local
    


def get_ckd_loss(cfg):
    name = cfg.TRAIN.CKD_LOSS
    content_level = cfg.TRAIN.CONTENT_LOSS_TYPE
    style_level = cfg.TRAIN.STYLE_LOSS_TYPE
    if name==None or name=="":
        name = CKD_loss.NAME

    if name==CKD_loss.NAME:
        return CKD_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_loss_Cov.NAME:
        return CKD_loss_Cov(content_level=content_level, style_level=style_level)
    elif name==CKD_GlobalLocal_soft_loss.NAME:
        return CKD_GlobalLocal_soft_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_GlobalLocal_hard_loss.NAME:
        return CKD_GlobalLocal_hard_loss(content_level=content_level, style_level=style_level)
    
    raise "error ckd loss type."