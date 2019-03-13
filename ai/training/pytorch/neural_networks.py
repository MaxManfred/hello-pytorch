import sys

import torch
import torch.nn as nn
import torch.optim as optim

from ai.training.pytorch.nn.le_net import LeNet
from ai.training.pytorch.utils.print_utils import print_all_examples


def example_nn00():
    print('NEURAL NETWORK')


def example_nn01():
    print('Create a LeNet and display layout')
    print(' ')

    le_net = LeNet()

    print(le_net)


def example_nn02():
    print('The learnable parameters of a model are returned by le_net.parameters()')
    print('')

    le_net = LeNet()
    params = list(le_net.parameters())

    print('Number of parameters = ', len(params))
    print('First layer (conv1) parameters (weights) shape', params[0].size())


def example_nn03():
    print('Processing inputs and calling backward')
    print('Let try a random 32x32 input.')
    print('Note: expected input size of this net (LeNet) is 32x32.')
    print('To use this net on MNIST dataset, please resize the images from the dataset to 32x32.')
    print(' ')

    le_net = LeNet()
    input_tensor = torch.randn(1, 1, 32, 32)
    print('input_tensor = torch.randn(1, 1, 32, 32)')

    output_tensor = le_net(input_tensor)
    print('output_tensor = le_net(input) = ', output_tensor)
    print(' ')

    print('Zero the gradient buffers of all parameters and backprop with random gradients')
    le_net.zero_grad()
    print('le_net.zero_grad()')

    output_tensor.backward(torch.randn(1, 10))
    print('output_tensor.backward(torch.randn(1, 10))')


def example_nn04():
    print('Computing loss')
    print('A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far '
          'away output is from the target.')
    print(' ')

    le_net = LeNet()

    input_tensor = torch.randn(1, 1, 32, 32)
    print('input_tensor = torch.randn(1, 1, 32, 32)')

    output_tensor = le_net(input_tensor)
    print('output_tensor = le_net(input) = ', output_tensor)
    print(' ')

    target_tensor = torch.randn(10)  # a dummy target, for example
    target_tensor = target_tensor.view(1, -1)  # make it the same shape as output_tensor
    print('target_tensor = torch.randn(10).view(1, -1) = ', target_tensor)
    print(' ')

    criterion = nn.MSELoss()
    print('criterion = nn.MSELoss()')

    loss_tensor = criterion(output_tensor, target_tensor)

    print('loss_tensor = criterion(output_tensor, target_tensor) = ', loss_tensor)
    print(' ')
    print('Now, if you follow loss in the backward direction, using its .grad_fn attribute, you will see a graph of '
          'computations that looks like this')
    print('input_tensor -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view -> linear -> relu -> '
          'linear -> relu -> linear -> MSELoss -> loss_tensor')


def example_nn05():
    print('Backprop loss')
    print('When we call loss_tensor.backward(), the whole graph is differentiated w.r.t. the loss, and all Tensors '
          'in the graph that has requires_grad = True will have their .grad Tensor accumulated with the gradient.')
    print('You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.')
    print('Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.')
    print(' ')

    le_net = LeNet()
    le_net.zero_grad()  # zeroes the gradient buffers of all parameters
    print('le_net.zero_grad()')
    print(' ')

    input_tensor = torch.randn(1, 1, 32, 32)
    output_tensor = le_net(input_tensor)
    target_tensor = torch.randn(10)
    target_tensor = target_tensor.view(1, -1)  # make it the same shape as output_tensor
    criterion = nn.MSELoss()
    loss_tensor = criterion(output_tensor, target_tensor)

    print('le_net.conv1.bias.grad before backward = ', le_net.conv1.bias.grad)

    loss_tensor.backward()

    print('le_net.conv1.bias.grad after backward = ', le_net.conv1.bias.grad)


def example_nn06():
    print('Parameters update')
    print('The simplest update rule used in practice is the Stochastic Gradient Descent (SGD)')
    print('weight = weight - learning_rate * gradient')
    print(' ')

    le_net = LeNet()

    # create your optimizer
    print()
    optimizer = optim.SGD(le_net.parameters(), lr=0.01)

    print('IN YOUR TRAINING LOOP:')
    print('\tGradient buffers had to be manually set to zero using optimizer.zero_grad().')
    print('\tThis is because gradients are accumulated as explained in Backprop section.')
    print('\toptimizer.zero_grad()')
    print(' ')
    optimizer.zero_grad()

    input_tensor = torch.randn(1, 1, 32, 32)
    output_tensor = le_net(input_tensor)
    target_tensor = torch.randn(10)
    target_tensor = target_tensor.view(1, -1)  # make it the same shape as output_tensor
    criterion = nn.MSELoss()
    loss_tensor = criterion(output_tensor, target_tensor)
    loss_tensor.backward()

    print('\tActually update by calling optimizer.step()')
    print('\toptimizer.step()')
    optimizer.step()  # Does the update


print_all_examples(sys.modules[__name__])
