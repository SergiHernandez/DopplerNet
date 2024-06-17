import os
from typing import List, Dict, Optional, Union, Tuple

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN, KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator

from nets.HeatmapGCN import HeatmapGCN
from nets.MSELossGCN import MSELossGCN
from nets.HeatmapRoIHeads import HeatmapRoIHeads
from nets.MSELossRoIHeads import MSELossRoIHeads

import albumentations as A

def load_freezed_model(weights_filename: Union[str,None] = None, is_gpu: bool = True,continue_train: bool = False) -> torch.nn.Module:
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights_filename is not None and os.path.exists(weights_filename):
        checkpoint = torch.load(weights_filename,map_location=device)
        print('freezed model epoch is %d' % checkpoint['epoch'])
        if 'cfg' in checkpoint:
            print(checkpoint['cfg'])
            cfg = checkpoint['cfg']
            model = load_model(cfg, is_gpu=is_gpu)
            model.load_state_dict(checkpoint['model_state_dict'])

    if continue_train == False:
        for name, p in model.named_parameters():
            #if "kpts_decoder" in name: #freeze image encoder part only
            p.requires_grad = False
    return model

def load_model(cfg, is_gpu: bool = False):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    backbone.out_channels = 1280



    if cfg.MODEL.NAME == "KeypointRCNN":

        # If we use the standard KeypointRCNN, we can load the default one with a pretrained backbone

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                       pretrained_backbone=True,
                                                                       num_classes=cfg.TRAIN.NUM_CLASSES,
                                                                       num_keypoints=cfg.TRAIN.NUM_KEYPOINTS,
                                                                       # WHEN KPT PREDICTOR IS SPECIFIED, we cannot put the num of kpts
                                                                       rpn_anchor_generator=anchor_generator)

    elif cfg.MODEL.NAME == "KeypointRCNN+GCN+HeatmapLoss":

        # If we use the KeypointRCNN+GCN with Heatmap loss, we initially load the same KeypointRCNN as before, but substitute the RoiHeads

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                       pretrained_backbone=True,
                                                                       num_classes=cfg.TRAIN.NUM_CLASSES,
                                                                       num_keypoints=cfg.TRAIN.NUM_KEYPOINTS,
                                                                       # WHEN KPT PREDICTOR IS SPECIFIED, we cannot put the num of kpts
                                                                       rpn_anchor_generator=anchor_generator)

        heatmap_roi_heads = HeatmapRoIHeads()

        model.roi_heads = heatmap_roi_heads

    elif cfg.MODEL.NAME == "KeypointRCNN+GCN+MSELoss":

        # If we use the KeypointRCNN+GCN with Heatmap loss, we initially load the same KeypointRCNN as before, but substitute the RoiHeads

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                       pretrained_backbone=True,
                                                                       num_classes=cfg.TRAIN.NUM_CLASSES,
                                                                       num_keypoints=cfg.TRAIN.NUM_KEYPOINTS,
                                                                       # WHEN KPT PREDICTOR IS SPECIFIED, we cannot put the num of kpts
                                                                       rpn_anchor_generator=anchor_generator)

        out_channels = model.backbone.out_channels

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)
        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, cfg.TRAIN.NUM_CLASSES)

        keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        keypoint_layers = tuple(512 for _ in range(8))
        keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

        keypoint_dim_reduced = 512  # == keypoint_layers[-1]
        keypoint_predictor = MSELossGCN(kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], num_kpts=cfg.TRAIN.NUM_KEYPOINTS, is_gpu=is_gpu)

        mse_roi_heads = MSELossRoIHeads(
            # Box
            box_roi_pool = box_roi_pool,
            box_head = box_head,
            box_predictor = box_predictor,
            fg_iou_thresh = 0.5,
            bg_iou_thresh = 0.5,
            batch_size_per_image = 512,
            positive_fraction = 0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            keypoint_roi_pool=keypoint_roi_pool,
            keypoint_head=keypoint_head,
            keypoint_predictor=keypoint_predictor
        )

        model.roi_heads = mse_roi_heads

    else:
        raise AssertionError(f"Model {cfg.MODEL.NAME} not implemented")


    
    print(model)

    model.output_type = 'img2kpts'

    return model


if __name__ == "__main__":
    from config.defaults import default_argument_parser, cfg_costum_setup
    args = default_argument_parser()
    cfg = cfg_costum_setup(args)   
    model = load_model(cfg, is_gpu=False)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    #print(model)