# Architectures used for teacher training and student compression.
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    # CNN teacher: two conv/BN/ReLU blocks with pooling, then a classifier.

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(1, 64), *block(64, 64), nn.MaxPool2d(2),       # 28 -> 14
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),   # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class StudentMLP(nn.Module):
    # Flattened image -> hidden layers -> class logits; sizes defaults to
    # 784-300-100-10. Each weight row corresponds to one output neuron.

    def __init__(self, sizes=(784, 300, 100, 10)):
        super().__init__()
        self.sizes = tuple(sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x):
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
