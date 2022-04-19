from detectron2.config import CfgNode as CN


def add_modelo1_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.MNV2 = CN()

    //_C.MODEL.MNV2.CONV_BODY = "V-39-eSE"
    _C.MODEL.MNV2.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.MNV2.NORM = "FrozenBN"

    _C.MODEL.MNV2.OUT_CHANNELS = 256

    _C.MODEL.MNV2.BACKBONE_OUT_CHANNELS = 256

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
