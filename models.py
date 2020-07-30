import torch
import torch.nn.functional as F

class AdviceModel(torch.nn.Module):
    def __init__(self):
        num_possible_actions = 7
        height = 60
        width = 64

        super().__init__()
        # print("HA")
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        self.lin1 = torch.nn.Linear((height - 4) * (width - 4) * 5, num_possible_actions)
        torch.nn.init.uniform_(self.lin1.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = xb.view(xb.size()[0], -1)
        xb = F.softmax(self.lin1(xb))  # use the formula for CNN shape!
        # print("LOL")
        return xb

'''
import models
class_ = getattr(models, "AdviceModel")
instance = class_()
'''