"""
Autograd: Automatic Differentiation
===================================

The ``autograd`` package provides automatic differentiation for all operations
on Tensors. It is a define-by-run framework, which means that your backprop is
defined by how your code is run, and that every single iteration can be
different.

Let us see this in more simple terms with some examples.

Tensor
--------

``torch.Tensor`` is the central class of the package. If you set its attribute
``.requires_grad`` as ``True``, it starts to track all operations on it. When
you finish your computation you can call ``.backward()`` and have all the
gradients computed automatically. The gradient for this tensor will be
accumulated into ``.grad`` attribute.

To stop a tensor from tracking history, you can call ``.detach()`` to detach
it from the computation history, and to prevent future computation from being
tracked.

To prevent tracking history (and using memory), you can also wrap the code block
in ``with torch.no_grad():``. This can be particularly helpful when evaluating a
model because the model may have trainable parameters with
``requires_grad=True``, but for which we don't need the gradients.

There’s one more class which is very important for autograd
implementation - a ``Function``.

``Tensor`` and ``Function`` are interconnected and build up an acyclic
graph, that encodes a complete history of computation. Each tensor has
a ``.grad_fn`` attribute that references a ``Function`` that has created
the ``Tensor`` (except for Tensors created by the user - their
``grad_fn is None``).

If you want to compute the derivatives, you can call ``.backward()`` on
a ``Tensor``. If ``Tensor`` is a scalar (i.e. it holds a one element
data), you don’t need to specify any arguments to ``backward()``,
however if it has more elements, you need to specify a ``gradient``
argument that is a tensor of matching shape.
"""
import sys

import torch

from ai.training.pytorch.utils.print_utils import print_all_examples


def example_ag00():
    print('AUTOGRAD')


def example_ag01():
    print('Create a tensor and set ``requires_grad=True`` to track computation with it')
    print('')

    x = torch.ones(2, 2, requires_grad=True)
    print(x)


def example_ag02():
    print('Do a tensor operation:')
    print('')
    print('y = x + 2')

    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2

    print(y)

    print('')
    print('y was created as a result of an operation, so it has a grad_fn')
    print('y.grad_fn = ', y.grad_fn)

    print('')
    print('Do more operations on y')
    z = y * y * 3
    print('z = y * y * 3')
    print(z)
    m = z.mean()
    print('m = z.mean()')
    print(m)


def example_ag03():
    print('.requires_grad_( ... ) changes an existing Tensor\'s requires_grad flag in-place.')
    print('The input flag defaults to False if not given.')
    print(' ')

    a = torch.randn(2, 2)
    print('a = torch.randn(2, 2)')
    a = ((a * 3) / (a - 1))

    print('a.requires_grad = ', a.requires_grad)
    print(' ')
    print('a.requires_grad_(True)')
    print('a.requires_grad = ', a.requires_grad)

    print(' ')

    b = (a * a).sum()
    print('b = (a * a).sum()')
    print('b.grad_fn = ', b.grad_fn)


def example_ag04():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    m = z.mean()

    print('Let\'s backprop now. Because m contains a single scalar, m.backward() is equivalent to '
          'm.backward(torch.tensor(1.)).')
    m.backward()

    print('Print gradients d(m)/dx through x.grad')
    print(' ')
    print(x.grad)


def example_ag05():
    # create a random 3x3 matrix and enable autograd
    x = torch.randn(3, requires_grad=True)

    # do some operations
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)

    # Now in this case y is no longer a scalar and torch.autograd could not compute the full Jacobian directly, but if
    # we just want the vector-Jacobian product, simply pass the vector to backward as argument

    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)


def example_ag06():
    # You can also stop autograd from tracking history on Tensors with .requires_grad=True by wrapping the code block in
    # with torch.no_grad():

    x = torch.randn(3, requires_grad=True)

    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)


print_all_examples(sys.modules[__name__])
