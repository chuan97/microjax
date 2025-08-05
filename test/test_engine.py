import torch

from microjax.engine import grad, relu


def test_sanity_check():
    def f_(x):
        z = 2 * x + 2 + x
        q = z + z * x
        h = relu(z * z)
        y = h + q + q * x

        return y

    grad_f = grad(f_)

    x = -4.0
    dfdx = grad_f(x)[0]

    X = torch.Tensor([x]).double()
    X.requires_grad = True

    Y = f_(X).value
    Y.backward()

    assert X.grad.item() == dfdx


def test_more_ops():
    def f_(a, b):
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + relu(b + a)
        d += 3 * d + relu(b - a)
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f

        return g

    grad_f = grad(f_)

    a = -4.0
    b = 2.0

    dfda, dfdb = grad_f(a, b)

    A = torch.Tensor([a]).double()
    B = torch.Tensor([b]).double()
    A.requires_grad = True
    B.requires_grad = True

    Y = f_(A, B).value
    Y.backward()

    assert A.grad.item() == dfda
    assert B.grad.item() == dfdb
