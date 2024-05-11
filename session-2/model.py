import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self,
                 h1=32, h2=64, h3=128, h4=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=64*14*14,
                out_features=1024
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=1024,
                out_features=num_clases
            ),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv2(x)
        bsz, nch, height, width = x.shape
        x = x.reshape(bsz, -1)
        y = self.mlp(x)
        return y
