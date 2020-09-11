import torch
import torch.nn.functional as F
import numpy as np

# if num_layers >= 3, it just fails since h_out and w_out go to 0... i either need larger images or a smaller conv kernel
# assuming default padding, stride, and dilations.
class AdviceModelGeneral(torch.nn.Module):
    def __init__(self, height, width, num_possible_actions, num_frames, num_layers):
        super().__init__()
        self.img_height = height
        self.img_width = width
        self.num_frames = num_frames
        self.num_layers = num_layers

        CONV_KERNEL_SIZE=3 # not sure if i'm 100% right here, will debug later
        MAXPOOL_KERNEL_SIZE=3 #same here

        self.conv_layers = []
        self.pooling_layers = []
        num_outputs = 0
        c=0
        h = self.img_height
        w = self.img_width
        d = self.num_frames # depth, for conv3d
        for i in range(num_layers):
            if num_frames == 1:
                # for some reason, if you don't assign a variable to a layer, the _modules attribute
                # of the model doesn't recognize it. maybe there's a way to track things as part of lists? which would
                # be more ideal... self._modules.add()?
                self.conv_layers.append(torch.nn.Conv2d(in_channels=CONV_KERNEL_SIZE ** i,
                                                   out_channels=CONV_KERNEL_SIZE ** (i + 1),
                                                   kernel_size=CONV_KERNEL_SIZE))
                self.conv_layers[i].to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                self.add_module("conv" + str(i), self.conv_layers[i])
                c = CONV_KERNEL_SIZE ** (i + 1)
                h = int(np.floor(h - (CONV_KERNEL_SIZE - 1) - 1 + 1))
                w = int(np.floor(w - (CONV_KERNEL_SIZE - 1) - 1 + 1))
                # todo: is this init still even good?
                torch.nn.init.uniform_(self.conv_layers[i].weight, a=-0.05, b=0.05)

                # self.pooling_layers.append(torch.nn.MaxPool2d(kernel_size=MAXPOOL_KERNEL_SIZE))
                # self.pooling_layers[i].to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                # self.add_module("maxpool" + str(i), self.pooling_layers[i])
                # h_in = int(np.floor(((h_out-MAXPOOL_KERNEL_SIZE-2)/MAXPOOL_KERNEL_SIZE)+1))
                # w_in = int(np.floor(((w_out - MAXPOOL_KERNEL_SIZE - 2) / MAXPOOL_KERNEL_SIZE) + 1))
            if num_frames >= 4:
                # https://heartbeat.fritz.ai/computer-vision-from-image-to-video-analysis-d1339cf23961
                self.conv_layers.append(torch.nn.Conv3d(in_channels=CONV_KERNEL_SIZE ** i,
                                                        out_channels=CONV_KERNEL_SIZE ** (i + 1),
                                                        kernel_size=(1,CONV_KERNEL_SIZE,CONV_KERNEL_SIZE)))
                self.conv_layers[i].to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                self.add_module("conv" + str(i), self.conv_layers[i])
                c = CONV_KERNEL_SIZE ** (i + 1)
                h = int(np.floor(h - (CONV_KERNEL_SIZE - 1) - 1 + 1))
                w = int(np.floor(w - (CONV_KERNEL_SIZE - 1) - 1 + 1))
                # d = int(np.floor(d - (CONV_KERNEL_SIZE - 1) - 1 + 1))
                torch.nn.init.uniform_(self.conv_layers[i].weight, a=-0.05, b=0.05)

        self.linear = torch.nn.Linear(h * w * c * d,num_possible_actions)
        torch.nn.init.uniform_(self.linear.weight, a=-0.05, b=0.05)

    def forward(self, xb):
        if self.num_frames == 1:
            xb = xb.reshape(1, 1, self.img_height, self.img_width)
        if self.num_frames >= 4:
            xb = xb.reshape(1, 1, self.num_frames, self.img_height, self.img_width)
        for i in range(self.num_layers):
            xb = F.relu(self.conv_layers[i](xb))
            # xb = self.pooling_layers[i](xb)
        xb = xb.view(xb.size()[0], -1)
        xb = torch.sigmoid(self.linear(xb))
        # xb = torch.softmax(self.linear(xb), dim=1)
        return xb

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
        # todo: softmax or sigmoid? softmax: all elements sum to 1.
        xb = torch.sigmoid(self.lin1(xb))  # use the formula for CNN shape!
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
