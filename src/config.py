import tomllib
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config.toml"
with CONFIG_FILE.open("rb") as f:
    data = tomllib.load(f)

device = data["main"]["device"]
default_save_dir = data["main"]["default_save_dir"]

dataset_dir = data["data"]["dataset_dir"]
im_size = data["data"]["image_size"]

batch_size = data["train"]["batch_size"]
epochs = data["train"]["epochs"]
learning_rate = data["train"]["learning_rate"]

loss_decimals = data["format"]["loss_decimals"]
percentage_decimals = data["format"]["percentage_decimals"]
time_decimals = data["format"]["time_decimals"]