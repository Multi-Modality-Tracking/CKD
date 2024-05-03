import torch
import torch.nn.functional as F
import cv2
import numpy as np

def recover_and_tomap(raw_map, global_index, vis='st', default_val=0., upsample=True):
    # raw_map: H,W
    # global_index: 1, N
    S_size = 256
    N = int(S_size**0.5)
    T_size = 64
    N_t = int(T_size**0.5)
    mid = T_size//2 if N_t%2 else T_size//2-N_t//2
    attn_map = torch.ones([S_size]).cuda()*default_val
    if vis=='st':
        attn_map[global_index.squeeze().long()] = raw_map[T_size:, mid]
    elif vis=='ts':
        attn_map[global_index.squeeze().long()] = raw_map[mid, T_size:]
    elif vis=='max_st':
        attn_map[global_index.squeeze().long()] = raw_map[T_size:, :T_size].max(-1).values
    elif vis=='mean_st':
        attn_map[global_index.squeeze().long()] = raw_map[T_size:, :T_size].mean(-1)
    attn_map = attn_map.reshape(N,N)

    if upsample:    
        attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')
    return attn_map


def attn_on_image(attn_map:np.ndarray, image:np.ndarray):
    # heat_map=np.uint8(255*attn_map)
    # heat_map=cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    # return heat_map*0.2+image
    heat_map=np.uint8(255*attn_map/attn_map.max())
    heat_map=cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)
    return heat_map*0.2+image*0.8
