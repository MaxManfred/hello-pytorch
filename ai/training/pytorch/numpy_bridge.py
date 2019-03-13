import sys

import numpy as np
import torch

from ai.training.pytorch.utils.print_utils import print_all_examples

"""
# NumPy Bridge
# ------------
#
# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
#
# The Torch Tensor and NumPy array will share their underlying memory
# locations, and changing one will change the other.
#
# Converting a Torch Tensor to a NumPy Array
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""


def example_npb00():
    print('NUMPY-PYTORCH BRIDGE')


###############################################################


def example_npb01():
    print('NumPy array from PyTorch tensor')
    print(' ')

    a = torch.ones(5)
    print('a = torch.ones(5) = ', a)

    b = a.numpy()
    print('b = a.numpy() = ', b)

    print(' ')
    print('See how changing the torch tensor changes the numpy array when calling a.add_(1)')
    a.add_(1)

    print('a = ', a)
    print('b = ', b)


###############################################################


def example_npb02():
    print('PyTorch tensor from NumPy array')
    print(' ')

    a = np.ones(5)
    print('a = np.ones(5) = ', a)

    b = torch.from_numpy(a)
    print('b = torch.from_numpy(a) = ', b)

    print(' ')
    print('See how changing the np array changed the torch tensor automatically when calling np.add(a, 1, out=a)')
    np.add(a, 1, out=a)

    print('a = ', a)
    print('b = ', b)


###############################################################


print_all_examples(sys.modules[__name__])
