from PIL import Image
import imageio
import os
from typing import Optional
import random
import numpy as np
import torch
import torch.nn.functional as F

def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:

    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

def gen_gif(src_dir):

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files in alphabetical order
    image_files.sort(key=lambda x: int(x.split(".")[0]))

    # Create an empty list to store the images
    images = []

    # Loop through the image files and append them to the list
    for image_file in image_files:
        image_path = os.path.join(src_dir, image_file)
        images.append(imageio.imread(image_path))

    # Set the output GIF file name
    output_file = os.path.join(src_dir, "../steps_imgs.gif")

    # Create the GIF from the list of images
    imageio.mimsave(output_file, images, duration=0.1)  # duration is the time between frames in seconds


def resize_net_attn_map(net_attn_maps, target_size):
    # net_attn_maps.size() = torch.Size([64,64])
    # F.interpolate requires input size to be torch.Size([bz, c, h, w]) so we need to do .unsqueeze(0) twice
    net_attn_maps = F.interpolate(
        net_attn_maps.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze().squeeze() # (1024,1024)
    return net_attn_maps

def attn_map_tr2np(attn_map):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    # image = Image.fromarray(normalized_attn_map)
    # image.save(save_path, format='PNG', compression=0)
    return normalized_attn_map
