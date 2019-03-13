import sys

from ai.training.pytorch.nn.simple_cnn import SimpleCNN
from ai.training.pytorch.utils.print_utils import print_all_examples


def example_nn01():
    simple_cnn = SimpleCNN()

    print(simple_cnn)


print_all_examples(sys.modules[__name__])
