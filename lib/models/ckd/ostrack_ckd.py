"""

"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224, resize_pos_embed
from .vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from greenlet import greenlet


class OSTrack_CKD(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, rgb_branch, tir_branch, teacher_rgb, teacher_tir, box_head, box_head_v, box_head_i, \
                 aux_loss=False, head_type="CORNER",mask_probability=0.0, mask_ratio=0.0, training=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.rgb_branch = rgb_branch
        self.tir_branch = tir_branch
        self.rgb_branch.mask_probability = mask_probability
        self.tir_branch.mask_probability = mask_probability
        self.rgb_branch.mask_ratio = mask_ratio
        self.tir_branch.mask_ratio = mask_ratio
        self.box_head = box_head
        if training:
            self.teacher_rgb = teacher_rgb
            self.teacher_tir = teacher_tir
            self.box_head_v = box_head_v
            self.box_head_i = box_head_i
        else:
            self.teacher_rgb = None
            self.teacher_tir = None
            self.box_head_v = None
            self.box_head_i = None

        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)



    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        if self.training:

            teacher_rgb_gr = greenlet(self.teacher_rgb)
            teacher_tir_gr = greenlet(self.teacher_tir)
            rgb_branch_gr = greenlet(self.rgb_branch)
            tir_branch_gr = greenlet(self.tir_branch)
            self.teacher_rgb.next_gr[0] = teacher_tir_gr
            self.teacher_tir.next_gr[0] = rgb_branch_gr
            self.rgb_branch.next_gr[0] = tir_branch_gr
            self.tir_branch.next_gr[0] = teacher_rgb_gr
            t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(z_li=template, x_li=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )
            
            t_x_tir, t_aux_dict_tir = teacher_tir_gr.switch()
                
            x_rgb, aux_dict_rgb = rgb_branch_gr.switch()
            
            x_tir, aux_dict_tir = tir_branch_gr.switch()
            
            aux_dict = {
                'x_rgb':x_rgb,
                'x_tir':x_tir,
                't_x_rgb':t_x_rgb,
                't_x_tir':t_x_tir,
                'aux_dict_rgb':aux_dict_rgb, 
                'aux_dict_tir':aux_dict_tir,
                'aux_dict_t_rgb':t_aux_dict_rgb,
                'aux_dict_t_tir':t_aux_dict_tir,}
            x = torch.cat([x_rgb, x_tir], 2)
            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            out = self.forward_head(feat_last, None)
            out_t_tir = self.forward_head(t_x_tir, None, head=self.box_head_i)
            out['out_t_tir'] = out_t_tir
            out_t_rgb = self.forward_head(t_x_rgb, None, head=self.box_head_v)
            out['out_t_rgb'] = out_t_rgb

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out
        
        else:
            rgb_branch_gr = greenlet(self.rgb_branch)
            tir_branch_gr = greenlet(self.tir_branch)
            self.rgb_branch.next_gr[0] = tir_branch_gr
            self.tir_branch.next_gr[0] = rgb_branch_gr
            x_rgb, aux_dict_rgb = rgb_branch_gr.switch(z_li=template[:2], x_li=search[:2],
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )
            
            x_tir, aux_dict_tir = tir_branch_gr.switch()
            
            aux_dict = {
                'aux_dict_rgb':aux_dict_rgb,
                'aux_dict_tir':aux_dict_tir}
            x = torch.cat([x_rgb,x_tir], 2)
            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out

    def forward_head(self, cat_feature, gt_score_map=None, head=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if head==None:
            box_head = self.box_head
        else:
            box_head = head
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack_ckd(cfg, training=True):
    patch_start_index = 1

    # RGB学生    
    rgb_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
    rgb_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # TIR学生
    if cfg.MODEL.SHARE_STUDENT:    
        tir_branch = rgb_branch
    else:
        tir_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        tir_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # RGB教师
    if cfg.MODEL.RGB_TEACHER:
        teacher_rgb = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_rgb.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_v = build_box_head(cfg, teacher_rgb.embed_dim)
    else:
        teacher_rgb = None
        box_head_v = None
    
    # TIR教师
    if cfg.MODEL.TIR_TEACHER:
        teacher_tir = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_tir.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_i = build_box_head(cfg, teacher_tir.embed_dim)
    else:
        teacher_tir = None
        box_head_i = None
    
    # 融合的跟踪头
    box_head = build_box_head(cfg, teacher_tir.embed_dim * 2)

    backbone_weight_filter = lambda param_dict : {k.replace("backbone.",""):v for k,v in param_dict.items() if 'backbone' in k}
    boxhead_weight_filter = lambda param_dict : {k.replace("box_head.",""):v for k,v in param_dict.items() if 'box_head' in k}
    def pos_embed_filter(param):
        param['backbone.pos_embed_z'] = resize_pos_embed(param['backbone.pos_embed_z'], posemb_new=torch.zeros(1,64,768),num_tokens=0)
        param['backbone.pos_embed_x'] = resize_pos_embed(param['backbone.pos_embed_x'], posemb_new=torch.zeros(1,256,768),num_tokens=0)
        param['backbone.pos_embed_z'] += param['backbone.temporal_pos_embed_z']
        param['backbone.pos_embed_x'] += param['backbone.temporal_pos_embed_x']
        return param
        if param['backbone.pos_embed_z'].shape[1]==144:
            param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'].reshape(1, 12, 12, 768)
            param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'][:, 2:-2, 2:-2, :].reshape(1, 64, 768)
            # param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'][:, :-4, :-4, :].reshape(1, 64, 768)
        if param['backbone.pos_embed_x'].shape[1]==576:
            param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'].reshape(1, 24, 24, 768)
            param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'][:, 4:-4, 4:-4, :].reshape(1, 256, 768)
            # param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'][:, :-8, :-8, :].reshape(1, 256, 768)
        return param

    if training:
        print("load RGB parameters:", cfg.MODEL.RGB_BRANCH)
        rgb_param = torch.load(cfg.MODEL.RGB_BRANCH, map_location="cpu")['net']
        if "DropTrack" in cfg.MODEL.RGB_BRANCH:
            rgb_param = pos_embed_filter(rgb_param)
        m,n = rgb_branch.load_state_dict(backbone_weight_filter(rgb_param), strict=False)
        print("missing keys: ", m)
        
        if not cfg.MODEL.SHARE_STUDENT:
            print("load TIR parameters:", cfg.MODEL.TIR_BRANCH)
            tir_param = torch.load(cfg.MODEL.TIR_BRANCH, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_BRANCH:
                tir_param = pos_embed_filter(tir_param)
            m,n = tir_branch.load_state_dict(backbone_weight_filter(tir_param), strict=False)
            print("missing keys: ", m)

        print("Tracking head type: concat")
        head_param = boxhead_weight_filter(rgb_param)
        for k,v in list(head_param.items()):
            if k in ['conv1_ctr.0.weight','conv1_offset.0.weight','conv1_size.0.weight']:
                head_param[k] = torch.cat([v,v],1)
        m,n = box_head.load_state_dict(head_param, strict=False)
        print("missing keys: ", m)

        if teacher_rgb!=None:
            print("load rgb teacher parameters:", cfg.MODEL.RGB_TEACHER)
            rgbTeacher_param = torch.load(cfg.MODEL.RGB_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.RGB_TEACHER:
                rgbTeacher_param = pos_embed_filter(rgbTeacher_param)
            m,n = teacher_rgb.load_state_dict(backbone_weight_filter(rgbTeacher_param), strict=False)
            print("missing keys: ", m)
        if box_head_v!=None:
            m,n = box_head_v.load_state_dict(boxhead_weight_filter(rgbTeacher_param), strict=False)
            print("missing keys: ", m)
        
        if teacher_tir!=None:
            print("load tir teacher parameters:", cfg.MODEL.TIR_TEACHER)
            tirTeacher_param = torch.load(cfg.MODEL.TIR_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_TEACHER:
                tirTeacher_param = pos_embed_filter(tirTeacher_param)
            m,n = teacher_tir.load_state_dict(backbone_weight_filter(tirTeacher_param), strict=False)
            print("missing keys: ", m)
        if box_head_i!=None:
            m,n = box_head_i.load_state_dict(boxhead_weight_filter(tirTeacher_param), strict=False)
            print("missing keys: ", m)



    model = OSTrack_CKD(
        rgb_branch,
        tir_branch,
        teacher_rgb,
        teacher_tir,
        box_head = box_head,
        box_head_v = box_head_v,
        box_head_i = box_head_i,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        mask_ratio=cfg.TRAIN.INPUT_MASK_RATIO,
        mask_probability=cfg.TRAIN.MASK_PROBABILITY,
        training = training,
    )
    if cfg.MODEL.PRETRAIN_FILE!="" and training:
        
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")['net'])
    return model