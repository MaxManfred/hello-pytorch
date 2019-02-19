import numpy as np
import torch

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
a = torch.ones(5)
print(a)

###############################################################
b = a.numpy()
print(b)

###############################################################
# See how the numpy array changed in value.

a.add_(1)
print(a)
print(b)

###############################################################
# Converting NumPy Array to Torch Tensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# See how changing the np array changed the Torch Tensor automatically

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###############################################################
# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
