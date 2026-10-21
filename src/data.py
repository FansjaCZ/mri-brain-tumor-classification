import kagglehub

from torchvision import datasets, transforms

from pathlib import Path

class Nickparvar():
    def __init__(self, path=str, im_size=[224,224]):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            #transforms.Normalize(),
        ])

        kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir=self.path)

    def get_train(self):
        return datasets.ImageFolder(
            root=self.path + "/Training",
            transform=self.transform,
        )

    def get_test(self):
        return datasets.ImageFolder(
            root=self.path + "/Testing",
            transform=self.transform,
        )



