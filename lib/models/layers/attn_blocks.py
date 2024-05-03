import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention, Attention_qkv


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template=None, global_index_search=None, mask=None, ce_template_mask=None, \
                keep_ratio_search=None, attn_ce=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)

        if global_index_template!=None:
            lens_t = global_index_template.shape[1]

            removed_index_search = None
            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                if attn_ce==None:
                    keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                    x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
                else:
                    keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                    x, global_index_search, removed_index_search = candidate_elimination(attn_ce, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, global_index_template, global_index_search, removed_index_search, attn
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x




class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            x_attn, attn = self.attn(self.norm1(x), mask, True)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class Attention_split(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def get_corrmap(self, q, k, mask):
        """
        输出原始的QK权重，未softmax
        """
        # q: B, N, C
        B, N1, C = q.shape
        B, N2, C = k.shape
        q = self.q_linear(q).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(k).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        return attn



    def get_attn_x(self, attn, v, return_attention=False, need_softmax=True):
        """
        根据attn权重，组织value
        """
        if need_softmax:        
            attn = attn.softmax(dim=-1)
            
        B,N,C = v.shape
        v = self.v_linear(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


    def general_forward(self, query, key, value, mask, return_attention=False):
        # q: B, N, C
        B, N1, C = query.shape
        B, N2, C = key.shape
        q = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(value).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
        

    def forward(self, q, k=None, v=None, mask=None, return_attention=False):
        k = q if k==None else k
        v = q if v==None else v
        return self.general_forward(q,k,v,mask, return_attention)



class MCEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                 next_gr:greenlet = None, layer_idx=-1):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.next_gr = next_gr

    def forward(self, x, global_index_template=None, global_index_search=None, mask=None, ce_template_mask=None, \
                keep_ratio_search=None, attn_li=()):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)

        if global_index_template!=None:
            lens_t = global_index_template.shape[1]

            removed_index_search = None
            if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
                keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                attn_li[1].append(attn)
                attn_li[2].append(self.layer_idx)
                branch_num = 4 if self.training else 2      # 先跑教师再跑学生
                if len(attn_li[1])==branch_num:             # 承上启下
                    max_attn = torch.stack(attn_li[1][-2:], dim=-1).max(-1).values
                    x, global_index_search, removed_index_search, topk_idx = candidate_elimination(max_attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
                    attn_li = ([global_index_search, removed_index_search, topk_idx], [], [])
                    self.next_gr[0].switch(attn_li)
                else:                                       # 处理结果并传递
                    attn_li = self.next_gr[0].switch(attn_li)
                    x, _, _, _ = candidate_elimination(None, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask, topk_idx=attn_li[0][-1])
                    global_index_search, removed_index_search = attn_li[0][0], attn_li[0][1]
                # if attn_ce==None:
                #     keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                #     x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
                # else:
                #     keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
                #     x, global_index_search, removed_index_search = candidate_elimination(attn_ce, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, global_index_template, global_index_search, removed_index_search, attn, attn_li
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

