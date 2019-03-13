import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ai.training.pytorch.nn.le_net import LeNet
from ai.training.pytorch.utils.print_utils import print_all_examples
from definitions import IMAGES_DATA_PATH


def example_cl00():
    print('TRAINING A CLASSIFIER')


def example_cl01():
    print('Specifically for vision, we have created a package called torchvision, that has data loaders for common '
          'datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., '
          'torchvision.datasets and torch.utils.data.DataLoader')
    print('For this tutorial, we will use the CIFAR10 dataset. It has the classes: \'airplane\', \'automobile\', '
          '\'bird\', \'cat\', \'deer\', \'dog\', \'frog\', \'horse\', \'ship\', \'truck\'. The images in CIFAR-10 '
          'are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.')

    print(' ')
    print('0. Select device to use')
    print(' ')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on ', device)

    print(' ')
    print('1. Load and normalize the CIFAR10 training and test data sets using torchvision')
    print(' ')

    training_set_loader, test_set_loader, classes = load_normalize()

    # get some random training images
    data_iterator = iter(training_set_loader)
    images, labels = data_iterator.next()

    # show images
    show_image(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    print(' ')
    print('2. Define a Convolutional Neural Network')
    print(' ')

    le_net = LeNet(number_input_channels=3, number_output_channels=16)
    le_net.to(device)
    print('LeNet created')

    print(' ')
    print('3. Define a loss function')
    print(' ')

    criterion = nn.CrossEntropyLoss()
    print('Loss function is ', criterion)
    optimizer = optim.SGD(le_net.parameters(), lr=0.001, momentum=0.9)
    print('Optimizer is', optimizer)

    print(' ')
    print('4. Train the network on the training data')
    print(' ')

    train(training_set_loader, le_net, optimizer, criterion, device)

    print(' ')
    print('5. Test the network on the test data')
    print(' ')

    # Okay, first step. Let us display an image from the test set to get familiar.
    data_iterator = iter(test_set_loader)
    images, labels = data_iterator.next()

    # print images
    show_image(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Okay, now let us see what the neural network thinks these examples above are
    outputs = le_net(images)
    # The outputs are energies for the 10 classes.
    # The higher the energy for a class, the more the network thinks that the image is of the particular class.
    # So, let’s get the index of the highest energy:

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # Let us look at how the network performs on the whole dataset.
    evaluate(test_set_loader, le_net, device)

    # What are the classes that performed well, and the classes that did not perform well
    evalulate_by_class(test_set_loader, le_net, classes, device)


def load_normalize():
    print('The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of '
          'normalized range [-1, 1].')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_set = torchvision.datasets.CIFAR10(root=IMAGES_DATA_PATH, train=True, download=True, transform=transform)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=IMAGES_DATA_PATH, train=False, download=True, transform=transform)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return training_set_loader, test_set_loader, classes


def show_image(img):
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(training_set_loader, neural_network, optimizer, criterion, device):
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_set_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = neural_network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def evaluate(test_set_loader, neural_network, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = neural_network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def evalulate_by_class(test_set_loader, neural_network, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in test_set_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = neural_network(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


print_all_examples(sys.modules[__name__])
