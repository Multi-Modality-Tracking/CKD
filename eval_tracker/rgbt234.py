"""
TODO: 填入结果文件的路径信息，评估跟踪器
"""

import rgbt     # pip install rgbt==1.0.1   github: https://github.com/opacity-black/RGBT_toolkit
from seqList import *
rgbt234 = rgbt.RGBT234()

result_path="your tracking result path"   # 
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)



rgbt234(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)

# rgbt234(
#     tracker_name="tracker_name2",
#     result_path=result_path,
#     seqs=seq_list
# )


# 画图
# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MPR, filename="eval_tracker/CAiATrack_RGBT234_MPR.png")
# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MSR, filename="eval_tracker/CAiATrack_RGBT234_MSR.png")

if __name__=="__main__":

    mpr_dict = rgbt234.MPR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "MPR", round(v[0]*100, 1))

    print('')
    msr_dict = rgbt234.MSR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "MSR", round(v[0]*100, 1))
