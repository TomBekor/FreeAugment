from typing import List

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def plot_tensors(tensor_list: List[torch.tensor], figsize=None):
    combined_tensors = torch.cat(tensor_list, dim=0).cpu()
    grid = make_grid(combined_tensors, nrow=2)  # Arrange in a row
    grid_image = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=figsize)
    plt.imshow(grid_image, aspect="auto")
    plt.axis("off")
    plt.show()
