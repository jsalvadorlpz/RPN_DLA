from detectron2.config import CfgNode as CN


def add_vovnet_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"

    _C.MODEL.VOVNET.OUT_CHANNELS = 256

    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

    _C = cfg

    _C.DATASETS.DLA = CN()

    _C.DATASETS.DLA.REGISTER_NEW_DATASET = False

    _C.DATASETS.DLA.TRAIN = CN()

    _C.DATASETS.DLA.TRAIN.NAME = "my_dataset_train"

    _C.DATASETS.DLA.TRAIN.JSON = "instances_train.json"

    _C.DATASETS.DLA.TRAIN.PATH = "./datasets/train/"

    _C.DATASETS.DLA.TEST = CN()

    _C.DATASETS.DLA.TEST.NAME = "my_dataset_test"

    _C.DATASETS.DLA.TEST.JSON = "instances_test.json"

    _C.DATASETS.DLA.TEST.PATH = "./datasets/test/"
