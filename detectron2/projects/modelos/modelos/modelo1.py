from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

def block1():
    # Block 1
    block1= nn.Sequential(
    nn.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'),
    nn.BatchNorm2d(64),
    nn.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
    nn.BatchNorm2d(64),
    nn.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
    )
    return block1
 
def block2():
    # Block 2
    block2= nn.Sequential(
    nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
    nn.BatchNorm2d(128),
    nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
    nn.BatchNorm2d(128),
    nn.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
    )
    return block2
  
  def block3():
    # Block 3
    block3= nn.Sequential(
    nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
    nn.BatchNorm2d(256),
    nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
    nn.BatchNorm2d(256),
    nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
    nn.BatchNorm2d(256),
    nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'),
    nn.BatchNorm2d(256),  
    nn.MaxPooling2D((2, 2), strides=(2, 2), name='stage3')
    )
    return block3
  
def block4():
    # Block 4
    block4= nn.Sequential(
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'),
    nn.BatchNorm2d(512),  
    nn.MaxPooling2D((2, 2), strides=(2, 2), name='stage4')
    )
    return block4
  
 def block5():
    # Block 5
    block5= nn.Sequential(
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
    nn.BatchNorm2d(512),
    nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'),
    nn.BatchNorm2d(512),  
    nn.MaxPooling2D((2, 2), strides=(2, 2), name='stage5')
    )
    return block5

class modelo1(Backbone):
   def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
     super().__init__()
     self.stem = stem
     self.num_classes = num_classes
     current_stride = self.stem.stride
     self._out_feature_strides = {"stem": current_stride}
     self._out_feature_channels = {"stem": self.stem.out_channels}
      
     
      
   
    x = GlobalMaxPooling2D()(x)
    
@BACKBONE_REGISTRY.register()    
def build_modelo1_backbone(cfg, input_shape):
    """
    Create a VoVNet instance from config.
    Returns:
        VoVNet: a :class:`VoVNet` instance.
    """
    out_features = cfg.MODEL.MODELO.OUT_FEATURES
    return Modelo1(cfg, input_shape.channels, out_features=out_features)

  
@BACKBONE_REGISTRY.register()
def build_modelo1_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_modelo1_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone  
  
