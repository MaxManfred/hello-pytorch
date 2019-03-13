import sys

import torch

from ai.training.pytorch.utils.print_utils import print_all_examples


def example_cu00():
    print('TENSORS ON CUDA')


def example_cu01():
    print('Tensors can be moved onto any device using the .to method.')
    print('Check if CUDA is available with torch.cuda.is_available()')
    print('Use torch.device(\'...\') to select device to put tensors to')
    print(' ')

    # We will use ``torch.device`` objects to move tensors in and out of GPU
    if torch.cuda.is_available():
        print('CUDA is available, hence get device')
        print(' ')
        device = torch.device('cuda')
        print('device = torch.device(\'cuda\')')

        x = torch.randn(4, 4, device=device)
        print('Directly create a tensor on GPU by x = torch.randn(4, 4, device=device)')
        x = torch.randn(4, 4)
        x = x.to(device)
        print('or create on CPU then move to GPU')
        print('x = torch.randn(4, 4)')
        print('x = x.to(device)')

        y = torch.ones_like(x, device=device)
        z = x + y
        print('Perform some computation on GPU')
        print('y = torch.ones_like(x, device=device)')
        print('z = x + y')

        z = z.to('cpu', torch.double)
        print('Get tensor back to CPU')
        print('z = z.to(\'cpu\', torch.double)')
        print('z = ', z)
        print('Notice that .to(...) can also change dtype')
    else:
        print('CUDA is not available, use CPU')
        print(' ')
        x = torch.randn(4, 4)
        print('x = torch.randn(4, 4)')
        y = torch.ones_like(x)
        z = x + y
        print('Perform some computation on CPU')
        print('y = torch.ones_like(x)')
        print('z = x + y')
        print('z = ', z)


print_all_examples(sys.modules[__name__])
