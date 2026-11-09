import kagglehub
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import filters

from pathlib import Path
import time
import math

SHARD_SIZE = 500
SHARDS_AT_A_TIME = 6

class BiasCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.corrector = sitk.N4BiasFieldCorrectionImageFilter()
    def forward(self, im_tensor):
        #if "dataset_size" in self:
        #    if not self.started_at or not self.looped:
        #        self.started_at = time.time()
        #        self.looped = 0
        #    elif self.looped == 10:
        #        done_at = datetime.datetime.now() + datetime.timedelta(seconds=self.dataset_size*((time.time()-self.started_at)/10))
        #        print(f"Preprocessing will be finished around {done_at.strftime("%H:%M:%S")}")
        #    self.looped += 1

        x = im_tensor[0].detach().contiguous().numpy()
        x = filters.n4(x, self.corrector)
        x = torch.tensor(x).unsqueeze(0)

        return x
        

class Nickparvar(Dataset):
    def __init__(self, path=str, im_size=[224,224], training=bool, mean=float, std=float):
        if training:
            self.dir = Path(path) / "training"
        else:
            self.dir = Path(path) / "testing"

        self.raw_transforms = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            BiasCorrection(),
        ])
        
        self.raw_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))

        if not self.dir.exists():
            
            self.dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()

            if training:
                dataset = datasets.ImageFolder(
                    root=self.raw_path / "Training",
                    transform=self.raw_transforms,
                )
            else:
                dataset = datasets.ImageFolder(
                    root=self.raw_path / "Testing",
                    transform=self.raw_transforms,
                )

            self.shard_save(dataset)

            torch.save({"classes" : dataset.classes, "size" : len(dataset)}, self.dir / "metadata.tar")

            hours, rest = divmod(time.time()-start_time, 3600)
            mins, secs = divmod(rest, 60)
            hours, mins, secs = round(hours), round(mins), round(secs)

            if training:
                print(f"Created training dataset cache in {hours}h {mins}m {secs}s")
            else:
                print(f"Created testing dataset cache in {hours}h {mins}m {secs}s")

        metadata = torch.load(self.dir / "metadata.tar")
        self.classes = metadata["classes"]
        self.size = metadata["size"]

        self._shards = []
        self._shards_num = []
        
        if training:
            self.run_time_transforms = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.Resize(im_size),
                transforms.Normalize([mean], [std]),
            ])
        else:
            self.run_time_transforms = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Normalize([mean], [std]),
            ])
        
        self.preprocess = transforms.Compose([
            self.raw_transforms,
            self.run_time_transforms,
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        shard_num = index // SHARD_SIZE
        shard_idx = index - shard_num * SHARD_SIZE
        if shard_num in self._shards_num:
            shard_cache_idx = self._shards_num.index(shard_num)
        else:
            shard_cache_idx = None

        if not isinstance(shard_cache_idx, int):
            shard_path = self.dir / f"shard_{shard_num:03d}.tar"
            shard = torch.load(shard_path)
            self._shards.append(shard)
            self._shards_num.append(shard_num)

            if len(self._shards) > SHARDS_AT_A_TIME:
                self._shards.pop(0)
                self._shards_num.pop(0)

        else:
            shard = self._shards[shard_cache_idx]

        if hasattr(self, "run_time_transforms"):
            return self.run_time_transforms(shard["images"][shard_idx]), shard["labels"][shard_idx]
        else:
            return shard["images"][shard_idx], shard["labels"][shard_idx]
        

    def shard_save(self, dataset):
        total_size = len(dataset)
        for start in range(0, total_size, SHARD_SIZE):
            end = min(start+SHARD_SIZE, total_size)
            samples = [dataset[i] for i in range(start, end)]
            torch.save({
                "images" : torch.stack([X for X, _ in samples]),
                "labels" : torch.tensor([y for _, y in samples]),
            }, self.dir / f"shard_{start // SHARD_SIZE:03d}.tar")
            print(f"Saved dataset {math.ceil(end / SHARD_SIZE)}/{total_size // SHARD_SIZE + 1}")



