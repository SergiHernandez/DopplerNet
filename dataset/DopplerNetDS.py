import os
from typing import Dict, List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as torch_transforms
from torchvision.transforms import functional as F

from PIL import Image
from torch.utils import data

# internal:

class DopplerNetDS(data.Dataset):
    def __init__(self, dataset_config, filenames_list: str=None, transform: A.core.composition.Compose = None):
        
        # Init super class
        super(DopplerNetDS, self).__init__()

        # Unpack dataset configs
        self.img_folder = dataset_config["img_folder"]
        self.anno_folder = dataset_config["anno_folder"]
        self.transform = transform
        self.input_size = dataset_config["input_size"]
        self.num_classes = dataset_config["num_classes"]
        self.NUM_KPTS = dataset_config["num_kpts"]
        self.ALLOWED_LABELS = dataset_config["allowed_kpts"]
        self.closed_contour = dataset_config["closed_contour"]
        print("Expected number of keypoints:", self.NUM_KPTS)
        print("Expected names of keypoints:", self.ALLOWED_LABELS)
        # get list of files in dataset:
        #self.create_img_list(filenames_list=filenames_list)

        # get kpts annotations
        #self.anno_dir = dataset_config["anno_folder"]
        #self.BOX_COORDS, self.LABELS = self.load_box_annotations(self.img_list)

        self.transform = transform
        aux_imgs_files = self.create_img_list(filenames_list)
        self.imgs_files = []
        self.annotations_files = []
        distinct_bbox_labels = set()

        # CHECK THAT THE SAMPLES HAVE THE REQUIRED LABELS
        allowed_labels_set = set(self.ALLOWED_LABELS)
        for fname in aux_imgs_files:
            valid_sample = True
            annotations_path = os.path.join(self.anno_folder, fname.replace("png", "npy"))
            data = np.load(annotations_path, allow_pickle=True)
            data = data.item()
            bboxes_original = data['bbox']
            bboxes_label = data['label']
            keypoints_original = data['kpts']
            kpts_labels_original = data['kpts_labels']
            for cycle_id in range(len(bboxes_original)):
                kpt_labels_set = set(kpts_labels_original[cycle_id].tolist())
                if len(allowed_labels_set.intersection(kpt_labels_set))!=len(allowed_labels_set):
                    valid_sample=False
            if valid_sample:
                self.imgs_files.append(fname)
                self.annotations_files.append(fname.replace("png", "npy"))
                distinct_bbox_labels.add(bboxes_label)
        print("Distinct labels", distinct_bbox_labels)
        self.label_map = {label: index for index, label in enumerate(distinct_bbox_labels)}
        print("Label mapping:", self.label_map)
        # Extras
        self.metadata_dir = self.img_folder.replace("frames", "metadata")
        self.COLORS = [(0, 0, 255), (255, 127, 0), (255, 70, 0), (0, 174, 174), (186, 0, 255), 
          (255, 255, 0), (204, 0, 175), (255, 0, 0), (0, 255, 0), (115, 8, 165), 
          (254, 179, 0), (0, 121, 0), (0, 0, 255)]

    # NO TOCAR!!!
    def create_img_list(self, filenames_list: str):
        """
        Called during construction. Creates a list containing paths to frames in the dataset
        """
        img_list_from_file = []
        with open(filenames_list) as f:
            img_list_from_file.extend(f.read().splitlines())
        #self.img_list = img_list_from_file
        return img_list_from_file


    def load_all_annotations(self, img_list: List) -> np.ndarray:
        """ Creates an array of annotated bounding box coordinates for the frames in the dataset. """
        BOX_COORDS = []
        LABELS = []
        if self.anno_folder is not None:
            for fname in img_list:
                annot_dir = np.load(os.path.join(self.anno_folder, fname.replace("png", "npy")), allow_pickle=True)
                annot_dir = annot_dir.item()
                BOX_COORDS.append(annot_dir['bbox'])
                LABELS.append(annot_dir['label'])
        return BOX_COORDS, LABELS


    def load_kpts_annotations(self, img_list: List) -> np.ndarray:
        """ Creates an array of annotated keypoints coorinates for the frames in the dataset. """
        KP_COORDS = []
        if self.anno_folder is not None:
            for fname in img_list:
                kpts = np.load(os.path.join(self.anno_folder, fname.replace("png", "npy")), allow_pickle=True)
                KP_COORDS.append(kpts[:,0:2])
                if len(self.LABELS) == 0 :
                    self.LABELS.append(kpts[:,2])
            #KP_COORDS = np.array(KP_COORDS).swapaxes(0, 1)

        return KP_COORDS

    def get_img_and_bxs(self, index: int):
        """
        Load and parse a single data point.
        Args:
            index (int): Index
        Returns:
            img (ndarray): RGB frame in required input_size
            kpts (ndarray): Denormalized, namely in img coordinates
            img_path (string): full path to frame file in image format (PNG or equivalent)
        """
        # ge paths:
        img_path = os.path.join(self.img_folder, self.img_list[index])
        
        # get image: (PRE-PROCESS UNIQUE TO UltraSound data)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #load boxes and convert them to tensors 
        boxes = self.BOX_COORDS[index].astype(int)

        #depends on multiclass and umbilical artery approach
        labels = [self.LABELS[index]] * len(boxes)
        #labels = [1] * len(boxes)

        boxes_length = len(boxes)
        #compute area of bbox
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        #define all objects as not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)

        data = {"img": img,
                "boxes": boxes,
                "img_path": img_path,
                "labels": labels,
                "area":area,
                "iscrowd":iscrowd, 
                "image_id":index + 1
                }
        return data

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.imgs_files[idx])
        annotations_path = os.path.join(self.anno_folder, self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        data = np.load(annotations_path, allow_pickle=True)
        data = data.item()
        bboxes_original = data['bbox']
        bboxes_label = data['label']
        keypoints_original = data['kpts']
        kpts_labels_original = data['kpts_labels']

        # PROCESS BBOX LABELS
        bboxes_labels = [self.label_map[bboxes_label]+1 for i in range(len(bboxes_original))]
        #bboxes_labels = [bboxes_label for i in range(len(bboxes_original))]

        # PROCESS KPTS AND KPT LABELS
        kpts = []
        kpts_labels = []
        for cycle_id in range(len(bboxes_original)):
            k = []
            l = []
            for kpt_id in range(len(keypoints_original[cycle_id])):
                if kpts_labels_original[cycle_id][kpt_id] in self.ALLOWED_LABELS:
                    k.append(keypoints_original[cycle_id][kpt_id] + [1]) # We add a 1 for the visibility
                    l.append(kpts_labels_original[cycle_id][kpt_id])
            assert len(k)==self.NUM_KPTS and len(l)==self.NUM_KPTS, "A sample doesnt have the approriate number of keypoints"
            kpts.append(k)
            kpts_labels.append(l)
        num_kpts = len(kpts[0])
        kpts_flat = [el[0:3] for kp in kpts for el in kp]  # we flatten the kpts to pass them to the transform function, but we unflatten them again later

        # TRANSFORM
        transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels, keypoints=kpts_flat)
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        bboxes_labels = transformed["bboxes_labels"]
        kpts = np.reshape(np.array(transformed['keypoints']), (-1,num_kpts,3)).tolist() # here we unflatten the kpts back

        # CONVERT TO TENSOR 
        image = F.to_tensor(image)
        tr = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
            ])
        image = tr(image)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes_labels = torch.tensor(bboxes_labels)
        kpts = torch.tensor(kpts, dtype=torch.float32)

        # PACK TARGETS
        targets = {}
        targets["boxes"] = bboxes
        targets["labels"] = bboxes_labels
        targets["keypoints"] = kpts
        targets["keypoint_labels"] = kpts_labels

        #print("IMAGE")
        #print(image.shape)
        #print("BBOX")
        #print(type(bboxes))
        #print(bboxes.dtype)
        #print(bboxes.shape)
        #print("BBOX LABELS")
        #print(type(bboxes_labels))
        #print(bboxes_labels.dtype)
        #print(bboxes_labels.shape)
        #print("KEYPOINTS")
        #print(type(kpts))
        #print(kpts.dtype)
        #print(kpts.shape)
        #print("")

        return image, targets, self.imgs_files[idx]
    
    def __len__(self):
        return len(self.imgs_files)