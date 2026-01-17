import torch
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        image_size: int = 64   # 👈 NEW
    ):
        super().__init__()

        # -------- Block 1 --------
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # -------- Block 2 --------
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # -------- Block 3 --------
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dropout = nn.Dropout(p=0.3)

        # -------- Dynamically compute flattened size --------
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, image_size, image_size)
            x = self.conv_block_1(dummy)
            x = self.conv_block_2(x)
            x = self.conv_block_3(x)
            flattened_size = x.view(1, -1).shape[1]

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
