import torch
import torchvision
import torchvision.transforms.functional as vision_F


def create_debug_grid(input_rows, max_views=4, normalize=False, imgsize=256):
    """
    Creates a 3xN grid for visualization.
    Row 1: Input views
    Row 2: Predicted target views
    Row 3: Ground truth target views
    """
    all_images_list = []
    for row in input_rows:
        black_image = torch.zeros(3, imgsize, imgsize).to(row.device)
        if normalize:
            # Denormalize all images from [-1, 1] to [0, 1]
            row = (row.clamp(-1, 1) + 1) / 2.0
        row = vision_F.resize(row, imgsize)
        row = list(row)
        num_views = len(row)
        stride = num_views // max_views
        if num_views > max_views:
            row = row[::stride][:max_views]
        while len(row) < max_views:
            row.append(black_image)
        all_images_list.extend(row)

    return torchvision.utils.make_grid(all_images_list, nrow=max_views)
