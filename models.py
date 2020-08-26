import torch
import torch.nn.functional as F
import numpy as np

class AdviceModel(torch.nn.Module):
    def __init__(self, height, width, num_possible_actions, num_frames):
        super().__init__()
        # NUM_FRAMES should be 1
        assert num_frames == 1, "Invalid value for num_frames"
        self.num_frames = num_frames
        self.img_height = height
        self.img_width = width
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3)
        self.lin1 = torch.nn.Linear(int(np.floor((((self.img_height - 4) - 2 -1)/3)+1)) *
                                    int(np.floor((((self.img_width - 4) - 2 -1)/3)+1)) *
                                    5,
                                    num_possible_actions)
        torch.nn.init.uniform_(self.lin1.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        xb = xb.reshape(1, 1, self.img_height, self.img_width)
        xb = F.relu(self.conv1(xb))
        xb = self.maxpool1(xb)
        xb = xb.view(xb.size()[0], -1)
        xb = F.softmax(self.lin1(xb), dim=1)  # use the formula for CNN shape!
        return xb

class AdviceModel2Layer(torch.nn.Module):
    def __init__(self, height, width, num_possible_actions, num_frames):
        super().__init__()
        # NUM_FRAMES should be 1
        assert num_frames == 1, "Invalid value for num_frames"
        self.num_frames = num_frames
        self.img_height = height
        self.img_width = width
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=25, kernel_size=5)
        torch.nn.init.uniform_(self.conv2.weight, a=-0.05, b=0.05)
        self.lin1 = torch.nn.Linear((self.img_height - 4 - 4) * (self.img_width - 4 - 4) * 25, num_possible_actions)
        torch.nn.init.uniform_(self.lin1.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        xb = xb.reshape(1, 1, self.img_height, self.img_width)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = xb.view(xb.size()[0], -1)
        xb = F.softmax(self.lin1(xb), dim=1)  # use the formula for CNN shape!
        return xb

class AdviceModel3d(torch.nn.Module):
    def __init__(self, height, width, num_possible_actions, num_frames):
        self.NUM_FRAMES = num_frames
        assert num_frames >= 4, "Invalid value for num_frames"
        # NUM_FRAMES should be 4 or more
        self.height = height
        self.width = width
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=5, kernel_size=3)
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        self.lin1 = torch.nn.Linear((height - 2) * (width - 2) * (self.NUM_FRAMES - 2) * 5, num_possible_actions)
        torch.nn.init.uniform_(self.lin1.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        xb = xb.reshape(1, 1, self.NUM_FRAMES, self.height, self.width)
        xb = F.relu(self.conv1(xb))
        xb = xb.view(xb.size()[0], -1)
        xb = F.softmax(self.lin1(xb), dim=1)  # use the formula for CNN shape!
        return xb
