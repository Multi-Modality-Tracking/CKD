import torch
import os,sys

# 确认参数是否冻结
def copyRight(file1, file2, key1='box_head.conv5_size.weight', key2=None):
    stateDict_1 = torch.load(file1, map_location="cpu")['net']
    stateDict_2 = torch.load(file2, map_location="cpu")
    for key1 in list(stateDict_2):
        # print(key1)
        try:
            a = torch.prod(torch.tensor((stateDict_1[key1].shape)))
        except:
            print(key1, 'shape不一致')
            return 
        key2 = key1
        b = (stateDict_1[key1]==stateDict_2[key2]).sum()
        if a==b:
            # print('参数已经冻结')
            continue
        else:
            print(key1, '参数不一致')

# 删除包里的梯度，节省空间
def delGrad(file):
    stateDict = torch.load(file, map_location="cpu")
    del stateDict['optimizer']
    torch.save(stateDict, file)


def qkv2q_k_v(file, new_file):
    # 将原来合在一起的qkv参数分开成q，k，v
    stateDict = torch.load(file, map_location="cpu")['net']
    stateDict_new = {}
    for k,v in list(stateDict.items()):
        if 'qkv' in k:
            print(f'transfer \"{k}\".')
            q_key = k.replace('qkv','q_linear')
            k_key = k.replace('qkv','k_linear')
            v_key = k.replace('qkv','v_linear')
            if 'weight' in k:
                stateDict[q_key] = v[:768, :]
                stateDict[k_key] = v[768:768*2, :]
                stateDict[v_key] = v[768*2:, :]
            elif 'bias' in k:
                stateDict[q_key] = v[:768]
                stateDict[k_key] = v[768:768*2]
                stateDict[v_key] = v[768*2:]
            del stateDict[k]
        # else:
        #     stateDict_new[k] = v
    torch.save(stateDict, new_file)


def param_anl(file):
    # adapter参数分析
    stateDict = torch.load(file, map_location="cpu")['net']
    for k,v in list(stateDict.items()):
        if 'adapt' in k:
            print(f"{k}, \tmean={v.mean()}, \tstd={v.std()}\n")


if __name__=="__main__":
    sys.path.append(os.getcwd())
    # copyRight(file1='/home/zhaojiacong/ostrack_promptFusion/output/checkpoints/train/ostrack_twobranch_prompt/vitb_256_mae_ce_32x4_ep300/OSTrack_twobranch_prompt_ep0025.pth.tar', 
    #           file2='/home/zhaojiacong/ostrack_rgbt/output/checkpoints/train/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300/OSTrack_twobranch_ep0084_q_k_v.pth.tar')
    # copyRight(file1 = "/home/zhaojiacong/tompnet_APF/checkpoints/ltr/tomp_apf/stage0/ToMPnet_ep0075.pth.tar", 
    #           file2 = "/home/zhaojiacong/tompnet_APF/checkpoints/ltr/tomp_apf/stage0/ToMPnet_ep0075.pth.tar", 
    #           key1='feature_extractor_rgb.layer3', key2='feature_extractor_tir.layer3')
    delGrad(file='/data/luandong/code/zhaojiacong/OSTrack_AlignBeforeFusion/output/checkpoints/train/ostrack_featureAlign/mse/OSTrack_twobranch_ep0020.pth.tar')
    # qkv2q_k_v(file="/home/zhaojiacong/ostrack_attnFusion/output/checkpoints/train/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300_True/OSTrack_twobranch_ep0010.pth.tar", 
            #    new_file="/home/zhaojiacong/ostrack_attnFusion/output/checkpoints/train/ostrack_twobranch/vitb_256_mae_ce_32x4_ep300_True/OSTrack_twobranch_ep0010_q_k_v.pth.tar")
    # param_anl(file='/home/zhaojiacong/ostrack_rgbt/output/checkpoints/train/adaption_net/v2/OSTrack_ep0035.pth.tar')