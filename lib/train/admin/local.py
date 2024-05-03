
from lib.utils.read_ip import get_host_ip


class EnvironmentSettings:
    def __init__(self):

        self.lmdb_dir="/home/zhaojiacong/datasets/lmdb_dataset/"
        
        self.workspace_dir = '.'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './tensorboard/'   # Directory for tensorboard files.
        self.wandb_dir = './wandb/'

        self.lasot_dir = '/home/liulei/Datasets/lasot/'
        self.got10k_dir = '/home/zhuyabin/dataset1/GOT/train/'
        self.trackingnet_dir = '/home/zhuyabin/dataset1/TrackingNet/'
        self.coco_dir = '/home/zhuyabin/dataset1/COCO2014/'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

        self.rgbt210_dir = '' # '/home/liulei/Datasets/RGBT210/'
        self.gtot_dir = '/home/liulei/Datasets/GTOT/'

        ip = get_host_ip()
        if ip=="172.17.122.103":
            self.lasher_dir = "/data/LasHeR/"
            self.lasher_trainingset_dir = "/data/LasHeR/"
            self.lasher_testingset_dir = "/data/LasHeR/"
            self.UAV_RGBT_dir= "/data/VTUAV/"
            self.rgbt234_dir = "/data/RGBT234/"
        elif ip in ["210.45.88.101", "172.17.122.101"]:
            self.lasher_dir = "/data/LasHeR/"
            self.lasher_trainingset_dir = "/data/LasHeR/"
            self.lasher_testingset_dir = "/data/LasHeR/"
            self.UAV_RGBT_dir= "/data/VTUAV/"
            self.rgbt234_dir = "/data/RGBT234/"
        elif ip=="10.10.1.98":
            self.lasher_dir = "/data1/Datasets/Tracking/LasHeR/"
            self.lasher_trainingset_dir = "/data1/Datasets/Tracking/LasHeR/"
            self.lasher_testingset_dir = "/data1/Datasets/Tracking/LasHeR/"
            self.UAV_RGBT_dir= "/data1/Datasets/Tracking/VTUAV/"
            self.rgbt234_dir = "/data1/Datasets/Tracking/RGBT234/"
        else:
            raise "需要添加数据集地址"
