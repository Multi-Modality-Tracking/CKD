
from lib.utils.read_ip import get_host_ip
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.lmdb_path="/home/zhaojiacong/datasets/lmdb_dataset/"
    
    ip = get_host_ip()
    if ip=="172.17.122.103":
        settings.lasher_path = "/data/LasHeR/"
        settings.vtuav_path= "/data/VTUAV/"
        settings.gtot_path = '/data/GTOT/'
        settings.rgbt210_path = "/data/RGBT210/"
        settings.rgbt234_path = "/data/RGBT234/"
        settings.gtot_dir = '/data/GTOT/'
    elif ip in ["210.45.88.101", "172.17.122.101"]:
        settings.lasher_path = "/data/LasHeR/"
        settings.vtuav_path= "/data/VTUAV/"
        settings.gtot_path = '/data/GTOT/'
        settings.rgbt210_path = "/data/RGBT210/"
        settings.rgbt234_path = "/data/RGBT234/"
        settings.gtot_dir = '/data/GTOT/'
    elif ip=="10.10.1.98":
        settings.lasher_path = "/data1/Datasets/Tracking/LasHeR/"
        settings.lashertestingSet_path = "/data1/Datasets/Tracking/LasHeR/"
        settings.vtuav_path= "/data1/Datasets/Tracking/VTUAV/"
        settings.gtot_path = '/data1/Datasets/Tracking/GTOT/'
        settings.rgbt210_path = "/data1/Datasets/Tracking/RGBT210/"
        settings.rgbt234_path = "/data1/Datasets/Tracking/RGBT234/"
        settings.gtot_dir = '/data1/Datasets/Tracking/GTOT/'
    else:
        raise "Add your dataset path !!!"
    
    settings.results_path = './tracking_result/'
    
    settings.save_dir="./output"
    settings.prj_dir = "."

    return settings