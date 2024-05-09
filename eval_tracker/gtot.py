"""
TODO: 填入结果文件的路径信息，评估跟踪器
"""

from rgbt.utils import RGBT_start
import rgbt     # pip install rgbt==1.0.1   github: https://github.com/opacity-black/RGBT_toolkit
from seqList import *
RGBT_start()
gtot = rgbt.GTOT()


result_path="your tracking result path"   # 
seq_list,length = where_seq_already(result_path, prefix="")
print("seq num: ", length)




gtot(
    tracker_name="tracker_name1",
    result_path=result_path,
    seqs=seq_list
)


# 画图
# gtot.draw_attributeRadar(metric_fun=gtot.MPR, filename="eval_tracker/CAiATrack_RGBT234_MPR.png")
# gtot.draw_attributeRadar(metric_fun=gtot.MSR, filename="eval_tracker/CAiATrack_RGBT234_MSR.png")

if __name__=="__main__":

    mpr_dict = gtot.MPR(seqs=seq_list)

    print('')
    for k,v in mpr_dict.items():
        print(k, "PR", round(v[0]*100, 1))

    print('')
    msr_dict = gtot.MSR(seqs=seq_list)

    for k,v in msr_dict.items():
        print(k, "SR", round(v[0]*100, 1))
