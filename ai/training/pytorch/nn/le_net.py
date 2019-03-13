import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, number_input_channels: int = 1, number_output_channels: int = 6):
        super(LeNet, self).__init__()

        # number_of_channels input image channel, number_output_channels output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(number_input_channels, number_output_channels, 5)
        self.conv2 = nn.Conv2d(number_output_channels, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
            You just have to define the forward function, and the backward function (where gradients are computed) is
            automatically defined for you using autograd. You can use any of the Tensor operations in the forward
            function.

            :param x: input tensor
            :return: output tensor
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
