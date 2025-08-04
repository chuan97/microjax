import torch

from microjax.engine import add, grad, mul, relu


def test_sanity_check():
    def f(x):
        z = add(add(mul(2, x), 2), x)
        q = add(relu(z), mul(z, x))
        h = relu(mul(z, z))
        y = add(h, add(q, mul(q, x)))

        return y

    grad_f = grad(f)

    x = -4.0
    dfdx = grad_f(x)[0]

    X = torch.Tensor([x]).double()
    X.requires_grad = True

    Y = f(X).value
    Y.backward()

    assert X.grad.item() == dfdx
