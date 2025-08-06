from dataclasses import dataclass
from typing import Callable

__all__ = [
    "Primitive",
    "Tracer",
    "relu",
    "grad",
    "value_and_grad",
    "trace",
    "backwards",
]

# ====== Primitive ops ======


class Primitive:
    """stores a single scalar function and its partial derivatives (gradient)"""

    def __init__(self, name: str, f: Callable, partials: list[Callable]):
        """
        f: (x_1, ..., x_n) -> y = f(x_1, ..., x_n)
        partials: df/dx_1, ..., df/dx_n
            with df/dx_i: (x_1, ..., x_n) -> y' = df/dx_i(x_1, ..., x_n)
        """
        self.name = name
        self.f = f
        self.partials = partials

    def __call__(self, *args) -> "Tracer":
        # convert to Tracer if needed
        args = [
            arg if isinstance(arg, Tracer) else Tracer(arg, parents=tuple(), op=None)
            for arg in args
        ]

        # compute output value
        out_val = self.f(*[arg.value for arg in args])

        # return output value as Tracer
        return Tracer(out_val, parents=tuple(args), op=self)

    def __repr__(self):
        return f"Primitive(name={self.name})"


relu = Primitive(
    name="relu",
    f=lambda x: x if x > 0 else 0,
    partials=[lambda x: 1.0 if x.value > 0.0 else 0.0],
)

_add = Primitive(name="add", f=lambda x, y: x + y, partials=[lambda x, y: 1] * 2)

_mul = Primitive(
    name="mul",
    f=lambda x, y: x * y,
    partials=[
        lambda x, y: y,
        lambda x, y: x,
    ],
)

_pow = Primitive(
    name="pow",
    f=lambda x, y: x**y,
    partials=[
        lambda x, y: y * x ** (y - 1) if y.value != 0.0 else 0.0,
        lambda x, y: 0.0,  # derivative w.r.t. exponent not supported
    ],
)


# ====== Tracer ======


@dataclass(frozen=True, eq=False)
class Tracer:
    """stores a single scalar value, the op that created it and its parents"""

    value: float
    parents: tuple["Tracer"]
    op: Primitive

    __hash__ = object.__hash__  # identity (not value) based hash

    def __add__(self, other) -> "Tracer":
        return _add(self, other)

    def __mul__(self, other) -> "Tracer":
        return _mul(self, other)

    def __pow__(self, other) -> "Tracer":
        return _pow(self, other)

    def __neg__(self) -> "Tracer":  # -self
        return self * -1

    def __radd__(self, other) -> "Tracer":  # other + self
        return self + other

    def __sub__(self, other) -> "Tracer":  # self - other
        return self + (-other)

    def __rsub__(self, other) -> "Tracer":  # other - self
        return other + (-self)

    def __rmul__(self, other) -> "Tracer":  # other * self
        return self * other

    def __truediv__(self, other) -> "Tracer":  # self / other
        return self * other**-1

    def __rtruediv__(self, other) -> "Tracer":  # other / self
        return self**-1 * other

    def __repr__(self):
        parent_vals = [p.value for p in self.parents]
        return f"<[Tracer: value={self.value}, parents={parent_vals}, op={self.op}]>"


# ===== Engine ======


def trace(f: Callable, *in_vals) -> tuple[Tracer, list[Tracer]]:
    """trace a function call (forward pass)"""
    # Trace inputs
    inputs = [
        Tracer(val, parents=(), op=None) if not isinstance(val, Tracer) else val
        for val in in_vals
    ]
    # Forward pass: ADG is built
    output = f(*inputs)

    # the case where f returns a constant
    if not isinstance(output, Tracer):
        output = Tracer(output, parents=(), op=None)

    return output, inputs


def backwards(output: Tracer) -> dict[Tracer, float]:
    """backpropagate gradients from output"""
    grads = {output: 1.0}

    for node in reverse_topo_sort(output):
        accum_grad = grads[node]

        for i, parent in enumerate(node.parents):
            partial = node.op.partials[i](*node.parents)

            grads[parent] = grads.get(parent, 0.0) + partial * accum_grad

    return grads


def grad(f: Callable) -> Callable:
    """given a scalar function return the function that computes its gradient"""

    def grad_f(*args):
        _, _, grads = evaluate_with_grad(f, *args)
        return grads

    return grad_f


def value_and_grad(f: Callable) -> Callable:
    """given a scalar function return a function that computes it and its gradient"""

    def value_and_grad_f(*args):
        output, _, grads = evaluate_with_grad(f, *args)
        return output, grads

    return value_and_grad_f


def evaluate_with_grad(f, *args):
    """
    given a scalar function and its arguments, run it forward tracing the inputs
    and return the output value and the gradient
    """
    output, inputs = trace(f, *args)
    grads = backwards(output)

    # are we inside an outer trace (higher order grads)
    inside_trace = any(isinstance(a, Tracer) for a in args)
    if not inside_trace:
        grads = {n: g.value if isinstance(g, Tracer) else g for n, g in grads.items()}
        output = output.value

    if len(inputs) == 1:
        grads = grads.get(inputs[0], 0.0)
    else:
        grads = [grads.get(inp, 0.0) for inp in inputs]

    return output, inputs, grads


def reverse_topo_sort(output: Tracer) -> list[Tracer]:
    """topological sort of the computational ADG"""
    visited = set()
    topo = []

    def build_topo(node):
        if node not in visited:
            visited.add(node)

            for parent in node.parents:
                build_topo(parent)

            topo.append(node)

    build_topo(output)
    return reversed(topo)
