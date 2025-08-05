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
    dfdx = grad_f(x)

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


def test_trivial_cases():
    def f_(x):
        return x

    grad_f = grad(f_)

    assert grad_f(0.0) == 1.0
    assert grad_f(4.0) == 1.0

    def g_(x):
        return 1.0

    grad_g = grad(g_)

    assert grad_g(0.0) == 0.0
    assert grad_g(-1.0) == 0.0

    def h_(x, y):
        return 1.0

    grad_h = grad(h_)

    assert grad_h(0.0, 1.0) == [0.0, 0.0]
    assert grad_h(-1.0, 5.0) == [0.0, 0.0]


def test_higher_order_grad():
    def f_(x):
        return x**2

    grad_grad_f = grad(grad(f_))

    assert grad_grad_f(0.0) == 2.0
    assert grad_grad_f(-5.0) == 2.0
