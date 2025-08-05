import torch

from microjax.engine import grad, relu


def test_sanity_check():
    def f(x):
        z = 2 * x + 2 + x
        q = z + z * x
        h = relu(z * z)
        y = h + q + q * x

        return y

    grad_f = grad(f)

    x = -4.0
    dfdx = grad_f(x)[0]

    X = torch.Tensor([x]).double()
    X.requires_grad = True

    Y = f(X).value
    Y.backward()

    assert X.grad.item() == dfdx
