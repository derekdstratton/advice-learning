import torch
import torch.nn.functional as F

# todo: consider having a parameter for AdviceModel that dictates the input shape for the model
# so that way the caller of AdviceModel can reshape based on this.
class AdviceModel(torch.nn.Module):
    def __init__(self, height, width, num_possible_actions):
        # num_possible_actions = 7
        # height = 60
        # width = 64

        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5)
        torch.nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        self.lin1 = torch.nn.Linear((height - 4) * (width - 4) * 5, num_possible_actions)
        torch.nn.init.uniform_(self.lin1.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = xb.view(xb.size()[0], -1)
        xb = F.softmax(self.lin1(xb), dim=1)  # use the formula for CNN shape!
        return xb
