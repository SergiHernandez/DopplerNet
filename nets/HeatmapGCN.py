import torch

from .gcn_spiral import GCN
from .HeatmapLayer import HeatmapLayerGaussian, HeatmapLayerL2

class HeatmapGCN (torch.nn.Module):
    
    def __init__(self, kpt_channels, gcn_channels, num_kpts=8, feature_size=2048, is_gpu=True, heatmap_layer = HeatmapLayerL2((32, 32))):

        super(HeatmapGCN, self).__init__()
        
        self.gcn = GCN(kpt_channels, gcn_channels, num_kpts=num_kpts, feature_size=feature_size, is_gpu=is_gpu)
        self.heatmap_layer = heatmap_layer

    def forward(self, features):

        # The GCN expects flattened vectors of features, needs flattening and probably pooling
        flat_features = features.flatten()
        relative_coordinates = self.gcn(features)
        heatmaps = self.heatmap_layer(relative_coordinates)

        return heatmaps



if __name__ == "__main__":

    heatmap_gcn = HeatmapGCN(kpt_channels=2, gcn_channels=[16, 32, 32, 48], num_kpts=8, is_gpu=False)
    print(heatmap_gcn)