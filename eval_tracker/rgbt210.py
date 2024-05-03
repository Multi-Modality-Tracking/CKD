"""
TODO: 填入结果文件的路径信息，评估跟踪器
"""

import rgbt     # pip install rgbt==1.0.1   github: https://github.com/opacity-black/RGBT_toolkit
from seqList import *
rgbt210 = rgbt.RGBT210()


result_path="your tracking result path"   # 
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)


rgbt210(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)

# rgbt210(
#     tracker_name="tracker_name2",
#     result_path=result_path,
#     seqs=seq_list
# )


# 画图
# rgbt210.draw_attributeRadar(metric_fun=rgbt210.MPR, filename="eval_tracker/CAiATrack_RGBT234_MPR.png")
# rgbt210.draw_attributeRadar(metric_fun=rgbt210.MSR, filename="eval_tracker/CAiATrack_RGBT234_MSR.png")

if __name__=="__main__":

    mpr_dict = rgbt210.PR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "PR", round(v[0]*100, 1))

    print('')
    msr_dict = rgbt210.SR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "SR", round(v[0]*100, 1))
