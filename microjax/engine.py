from dataclasses import dataclass
from typing import Callable


class Primitive:
    def __init__(self, name: str, forward_fn: Callable, backward_fns: list[Callable]):
        """
        forward_fn: f: (x_1, ..., x_n) -> y = f(x_1, ..., x_n)
        backward_fns: df/dx_1, ..., df/dx_n
            with df/dx_i: (x_1, ..., x_n) -> y' = df/dx_i(x_1, ..., x_n)
        """
        self.name = name
        self.forward_fn = forward_fn
        self.backward_fns = backward_fns

    def __call__(self, *args):
        # convert to Tracer if needed
        args = [
            arg if isinstance(arg, Tracer) else Tracer(arg, parents=tuple(), op=None)
            for arg in args
        ]

        # compute output value
        out_val = self.forward_fn(*[arg.value for arg in args])

        # return output value as Tracer
        return Tracer(out_val, parents=tuple(args), op=self)

    def __repr__(self):
        return f"Primitive(name={self.name})"


@dataclass(frozen=True)
class Tracer:
    value: float
    parents: tuple["Tracer"]
    op: Primitive


def trace(f: Callable, *in_vals: list[float]):
    # Trace inputs
    inputs = [Tracer(val, parents=tuple(), op=None) for val in in_vals]
    # Forward pass: ADG is built
    output = f(*inputs)
    return output, inputs


def backwards(output: Tracer):
    grads = {output: 1.0}

    for node in reverse_topo_sort(output):
        prev_grad = grads[node]
        for i, parent in enumerate(node.parents):
            partial = node.op.backward_fns[i](*[p.value for p in node.parents])
            grads[parent] = grads.get(parent, 0.0) + partial * prev_grad

    return grads


def grad(f: Callable):
    def grad_f(*args):
        output, inputs = trace(f, *args)
        grads = backwards(output)
        return [grads[input] for input in inputs]

    return grad_f


def reverse_topo_sort(output: Tracer):
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


# ---------- basic operations ----------
add = Primitive(
    name="add", forward_fn=lambda x, y: x + y, backward_fns=[lambda x, y: 1] * 2
)

mul = Primitive(
    name="mul",
    forward_fn=lambda x, y: x * y,
    backward_fns=[
        lambda x, y: y,
        lambda x, y: x,
    ],
)

relu = Primitive(
    name="relu", forward_fn=lambda x: x if x > 0 else 0, backward_fns=[lambda x: x > 0]
)
