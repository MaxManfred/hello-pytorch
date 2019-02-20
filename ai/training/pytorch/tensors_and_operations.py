"""
Tensors
^^^^^^^

Tensors are similar to NumPy’s ndarrays, with the addition being that
Tensors can also be used on a GPU to accelerate computing.
"""

from __future__ import print_function

import inspect
import sys

import torch

from ai.training.pytorch.utils.print_utils import print_all_examples


def example_tb00():
    print('TENSOR BASICS')


def example_tb01():
    print('Construct a 5x3 matrix, uninitialized:')
    print('')
    x = torch.empty(5, 3)
    print(x)


def example_tb02():
    print('Construct a randomly initialized matrix:')
    print('')
    x = torch.rand(5, 3)
    print(x)


def example_tb03():
    print('Construct a matrix filled zeros and of dtype long:')
    print('')
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)


def example_tb04():
    print(inspect.cleandoc("""
        Construct a tensor directly from data or create a tensor based on an existing tensor.
        These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user"""))
    print('')
    x = torch.tensor([5.5, 3])
    print(x)
    print('')

    x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
    print(x)
    print('')

    x = torch.randn_like(x, dtype=torch.float)  # override dtype!
    print(x)  # result has the same size
    print('')


def example_tb05():
    print('Get tensor size:')
    print('')
    x = torch.rand(5, 3, dtype=torch.float)
    print(x)
    print('')

    print(x.size())
    print('')
    print('Note: torch.Size is in fact a tuple, so it supports all tuple operations.')


def example_to00():
    print('TENSOR OPERATIONS')


def example_to01():
    print(inspect.cleandoc("""
        There are multiple syntaxes for operations.
        In the following example, we will take a look at the addition operation."""))
    print('')

    x = torch.rand(5, 3)
    print('x = ', x)
    print('')

    y = torch.rand(5, 3)
    print('y = ', y)
    print('')

    print('Addition: syntax 1')
    print('')

    print('x + y = ', x + y)
    print('')

    print('Addition: syntax 2')
    print('')

    print('torch.add(x, y) = ', torch.add(x, y))
    print('')

    print('Addition: providing an output tensor as argument')
    print('')

    result = torch.empty(5, 3)
    torch.add(x, y, out=result)
    print('torch.add(x, y, out=result) = ', result)
    print('')

    print('Addition: in-place')
    print('')

    y.add_(x)
    print('y.add_(x) = ', y)
    print('')

    print(inspect.cleandoc("""
        Note: any operation that mutates a tensor in-place is post-fixed with an ``_``.
        For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``."""))
    print('')


def example_to02():
    print('You can use standard NumPy-like indexing with all bells and whistles!')
    print('')

    x = torch.rand(5, 3)
    print('x = ', x)
    print('')

    print('x[:, 1] = ', x[:, 1])


def example_to03():
    print('Resizing: If you want to resize/reshape tensor, you can use ``torch.view``:')
    print('')

    x = torch.randn(4, 4)
    print('x = ', x)
    print('')

    y = x.view(16)
    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

    print('x.size() = ', x.size())
    print('y.size() = ', y.size())
    print('z.size() = ', z.size())


def example_to04():
    print('If you have a one element tensor, use ``.item()`` to get the value as a Python number:')
    print('')

    x = torch.randn(1)
    print('x = ', x)
    print('x.item() = ', x.item())


print_all_examples(sys.modules[__name__])
