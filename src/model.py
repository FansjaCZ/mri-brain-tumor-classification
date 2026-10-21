from torch import nn

class Model(nn.Module):
    def __init__(self, im_size):
        super().__init__()
        post_im_size = (im_size[0]/(2**5), im_size[1]/(2**5))
        self.softmax = nn.Softmax(1)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, groups=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.global_avarage_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        #self.block_dense1 = nn.Linear(int(256*post_im_size[0]*post_im_size[1]), 512)
        self.block_dense1 = nn.Linear(256, 512)
        self.block_dense2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.005),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_avarage_pooling_2d(x)
        x = self.flatten(x)
        x = self.block_dense1(x)
        x = self.block_dense2(x)

        if not self.training:
            x = self.softmax(x)
        return x