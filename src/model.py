from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 512, 7, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )

        #self.global_avarage_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 4)

        self.apply_regularisation = [self.fc1, self.fc2, self.fc3, self.fc4]

        self.softmax = nn.Softmax(1)
        self.use_softmax = False

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        #x = self.global_avarage_pooling_2d(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.ReLU(x)
        if self.training:
            x = self.dropout(x)

        x = self.fc2(x)
        x = self.ReLU(x)
        if self.training:
            x = self.dropout(x)

        x = self.fc3(x)
        x = self.ReLU(x)
        if self.training:
            x = self.dropout(x)
            
        x = self.fc4(x)

        if self.use_softmax:
            x = self.softmax(x)
        return x