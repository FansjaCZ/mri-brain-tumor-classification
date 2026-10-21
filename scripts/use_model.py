import torch
from torchvision.transforms import transforms

from PIL import Image

from model import Model
from data import Nickparvar
from train import get_sate_dict
import config

from pathlib import Path
import time
import sys

def main(im_path=str, save_path=None):

    if not save_path:
        save_path = config.default_save_dir

    if not Path(save_path).exists():
        print("Training save path (argument 1) does not exist")
        return

    if not Path(im_path).exists():
        print("Image path (argument 2) does not exist")
        return

    im = Image.open(im_path)

    dataset = Nickparvar(config.dataset_dir, config.im_size)
    classes = dataset.get_test().classes

    im_tensor = dataset.transform(im).unsqueeze(0).to(config.device)

    model = Model(config.im_size)
    model.to(config.device)
    model.load_state_dict(get_sate_dict(save_path))

    print(f"Loaded training from {save_path}")
    
    time_start = time.time()

    model.eval()
    with torch.no_grad():
        out = model(im_tensor)

    print(f"The input was classified as {classes[out.argmax(1).item()]} ({round(time.time()-time_start, config.time_decimals)}s)")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Not enough arguments")