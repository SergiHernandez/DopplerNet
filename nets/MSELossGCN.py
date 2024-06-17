import torch
from typing import List, Dict, Optional, Union, Tuple

from .gcn_spiral import GCN

class MSELossGCN (torch.nn.Module):
    
    def __init__(self, kpt_channels, gcn_channels, num_kpts=8, feature_size=6272, is_gpu=True):

        super(MSELossGCN, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.gcn = GCN(kpt_channels, gcn_channels, num_kpts=num_kpts, feature_size=feature_size, is_gpu=is_gpu)
        self.num_kpts = num_kpts

    def forward(self, features):

        # The GCN expects flattened vectors of features, needs flattening and probably pooling

        if features.shape[0] == 0: # If there are bounding boxes in the batch, decode their keypoints
            return torch.zeros(features.shape[0], self.num_kpts, 2, dtype=features.dtype, device=features.device, requires_grad=True)

        #print(features.shape)
        reduced_features = self.conv(features)
        #print(reduced_features.shape)
        flat_features = reduced_features.flatten(start_dim=1)
        #print(flat_features.shape) # should be of size ([B, 4096]), in other words, for each image in the batch, a vector of 4096 features
        relative_coordinates = self.gcn(flat_features)
        #print(relative_coordinates.shape) # Here the size is, and should be, ([B, K, 2])

        #print(relative_coordinates)

        return relative_coordinates



if __name__ == "__main__":

    heatmap_gcn = MSELossGCN(kpt_channels=2, gcn_channels=[16, 32, 32, 48], num_kpts=8, is_gpu=False)
    print(heatmap_gcn)