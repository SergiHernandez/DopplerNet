import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple

from dataset.DopplerNetDS import DopplerNetDS
from utils.utils_files import AverageMeter, to_numpy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time 
########################################################
########################################################
# Run single epoch for Train/Validate/Eval
########################################################
########################################################

def train_vanilla(epoch: int,
                  loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  model: torch.nn.Module,
                  device: torch.device,
                  criterion: torch.nn.Module,
                  prossesID: Union[str,None] = None
                  ) -> dict[str, AverageMeter]:

    model.train()
    prefix = 'Training'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = {"main": AverageMeter(),
              "kpt_loss": AverageMeter(),
              "bbox_loss": AverageMeter(),
              "class_loss": AverageMeter(),
              "rpn_loss": AverageMeter(),
              "object_loss": AverageMeter()}
    
    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):
            model.train()
            # ================= Extract Data ==================
            images = data[0]
            targets = data[1]
            filenames = data[2]

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # =================== forward =====================
            #batch_loss, batch_output = run_forward(model, data, criterion, device)
            loss_dict = model(images, targets)
            sum_losses = sum(loss for loss in loss_dict.values())
            print(loss_dict)
            # =================== backward ====================
            if optimizer is not None:

                #a = model.roi_heads.keypoint_predictor.gcn.kpts_decoder.decoder_layers[1].layer.weight.clone().detach()
                optimizer.zero_grad()
                sum_losses.backward()
                optimizer.step()

                #print(torch.equal(a, model.roi_heads.keypoint_predictor.gcn.kpts_decoder.decoder_layers[1].layer.weight))
                #print("weights shape", a.shape)
                #print(a)
                #for i in range(a.shape[0]):
                #    for j in range(a.shape[1]):
                #        for k in range(a.shape[2]):
                #            for l in range(a.shape[3]):
                #        if a[i, j] != model.roi_heads.keypoint_predictor.gcn.kpts_decoder.decoder_layers[1].layer.weight[i, j]:
                #            print(f"{a[i, j]} != {model.roi_heads.keypoint_predictor.gcn.kpts_decoder.decoder_layers[1].layer.weight[i, j]}")
            pbar.update()

            # SERGI: to compute the different custom losses, it is necessary to put the model in eval mode and do inference
            # because in training the model does not return predictions
            # accumulate losses:
            losses["main"].update(loss_dict["loss_keypoint"].cpu().item(), len(filenames))
            losses["kpt_loss"].update(loss_dict["loss_keypoint"].cpu().item(), len(filenames))
            losses["bbox_loss"].update(loss_dict["loss_box_reg"].cpu().item(), len(filenames)) #???????????
            losses["class_loss"].update(loss_dict["loss_classifier"].cpu().item(), len(filenames))
            losses["rpn_loss"].update(loss_dict["loss_rpn_box_reg"].cpu().item(), len(filenames))
            losses["object_loss"].update(loss_dict["loss_objectness"].cpu().item(), len(filenames))
            # losses returned by model: {'loss_classifier', 'loss_box_reg', 'loss_keypoint', 'loss_objectness', 'loss_rpn_box_reg'}
            # losses expected by tensorboard outside this file: {'main', 'bbox_loss', 'class_loss', 'rpn_loss', 'object_loss'}

    return losses


def validate(mode: str,
             epoch: int,
             loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             criterion: torch.nn.Module,
             prossesID: Union[str,None] = None
             ) -> Tuple[int, list]:
    """validation"""

    model.eval()
    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    inputs, outputs = dict(), dict()
    losses = {"main": AverageMeter(),
              "kpt_loss": AverageMeter(),
              "bbox_loss": AverageMeter(),
              "class_loss": AverageMeter(),
              "rpn_loss": AverageMeter(),
              "object_loss": AverageMeter()}

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            images = data[0]
            targets = data[1]
            filenames = data[2]

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # =================== forward =====================
            output = model(images)

            pbar.update()


    return output

########################################
########################################
# Sample dataset
########################################
########################################
def sample_dataset(trainset: DopplerNetDS, valset: DopplerNetDS, testset: DopplerNetDS, overfit: bool, batch_size: int = 8, num_workers: int = 8):
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        annotated = np.random.choice(len(trainset), num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                  shuffle=False, pin_memory=True, collate_fn=collate_fn)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                 shuffle=False, pin_memory=True, collate_fn=collate_fn)
        print("DATA: Sampling identical sets of {} ANNOTATED examples for train and val sets.. ".format(num_samples_for_overfit))

    else:
        # --- Train: ---
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers, collate_fn=collate_fn)
        # --- Val: ---
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=True,
                                                     num_workers=num_workers, collate_fn=collate_fn)

    # --- Test: ---
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers, collate_fn=collate_fn)
    else:
        testloader = []

    return trainloader, valloader, testloader



def collate_fn(batch):
    return tuple(zip(*batch))