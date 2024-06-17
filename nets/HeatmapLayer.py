import torch
import math

# IMPORTANT: only works for keypoints in 2D
# TODO: make it work for points in any dimensions
class HeatmapLayerL2(torch.nn.Module):
    def __init__(self, heatmap_shape):
        super(HeatmapLayerL2, self).__init__()
        self.heatmap_shape = heatmap_shape
        gamma = 1.0
        for dim in heatmap_shape:
            gamma *= (dim**2)
        self.gamma = torch.nn.Parameter(torch.tensor(1/gamma), requires_grad=True)

    def forward(self, kpts):

        B = kpts.shape[:-1]
        axes = kpts.shape[-1] # By now it only works with 2D kpts
        assert axes == 2, f"Keypoints in {axes}D were inputed. HeatmapLayer is only implemented to work with 2D kpts by now."
        device = kpts.device

        indices = []
        for axis in range(axes):
            indices.append(torch.arange(self.heatmap_shape[axis], device=device).unsqueeze(axes-axis-1).expand(*B, *self.heatmap_shape))

        indices_tensor = torch.stack(indices, dim=len(B))

        coords_tensor_squares = kpts
        for axis in range(axes):
            coords_tensor_squares = coords_tensor_squares[..., None]
            coords_tensor_squares= coords_tensor_squares.expand(*(coords_tensor_squares.shape[:-1]), self.heatmap_shape[axis])
            
        i_th_dimensional_distances = torch.sub(indices_tensor, coords_tensor_squares)
        i_th_dimensional_distances_squared = i_th_dimensional_distances.pow(2)
        summed_distances = i_th_dimensional_distances_squared.sum(dim=2)
        almost_normalized = torch.mul(summed_distances, self.gamma)
        if almost_normalized.numel() > 0:
            min_p = almost_normalized.min()
            max_p = almost_normalized.max()
            scale = 1/(max_p-min_p)
            almost_normalized = (almost_normalized-min_p) * scale
        return -almost_normalized
    

class HeatmapLayerGaussian(torch.nn.Module):
    def __init__(self, heatmap_shape, sigma=None):

        super(HeatmapLayerGaussian, self).__init__()
        self.heatmap_shape = heatmap_shape
        if sigma==None:
            self.sigma = heatmap_shape[0]
        else:
            self.sigma = sigma


    def forward(self, kpts):

        B = kpts.shape[:-1]
        axes = kpts.shape[-1] # By now it only works with 2D kpts
        assert axes == 2, f"Keypoints in {axes}D were inputed. HeatmapLayer is only implemented to work with 2D kpts by now."
        device = kpts.device
        sigma = self.sigma
        
        indices = []
        for axis in range(axes):
            indices.append(torch.arange(self.heatmap_shape[axis], device=device).unsqueeze(axes-axis-1).expand(*B, *self.heatmap_shape))

        indices_tensor = torch.stack(indices, dim=len(B))

        coords_tensor_squares = kpts

        
        for axis in range(axes):
            coords_tensor_squares = coords_tensor_squares[..., None]
            coords_tensor_squares = coords_tensor_squares.expand(*(coords_tensor_squares.shape[:-1]), self.heatmap_shape[axis])

        i_th_dimensional_distances = torch.sub(indices_tensor, coords_tensor_squares)
        i_th_dimensional_distances = i_th_dimensional_distances.pow(2)
        i_th_dimensional_distances = i_th_dimensional_distances.sum(dim=2)
        i_th_dimensional_distances = i_th_dimensional_distances * (1/sigma)
        i_th_dimensional_distances = i_th_dimensional_distances.pow(2)
        i_th_dimensional_distances = -(1/2) * i_th_dimensional_distances
        #i_th_dimensional_distances = torch.exp(i_th_dimensional_distances)
        
        i_th_dimensional_distances = (1/self.sigma * math.sqrt(2*math.pi))*i_th_dimensional_distances
        if i_th_dimensional_distances.numel() > 0:
            min_p = i_th_dimensional_distances.min()
            max_p = i_th_dimensional_distances.max()
            scale = 1/(max_p-min_p)
            i_th_dimensional_distances = (i_th_dimensional_distances-min_p) * scale

        return i_th_dimensional_distances