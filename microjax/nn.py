import random
from dataclasses import dataclass
from typing import Union

from .engine import Tracer, relu

Scalar = Union[float | Tracer]


@dataclass(frozen=True)
class Neuron:
    w: list[Scalar]
    b: Scalar = 0.0
    nonlin: bool = True

    def __call__(self, x: list[Scalar]) -> Scalar:
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return relu(act) if self.nonlin else act

    def parameters(self) -> list[Scalar]:
        return self.w + [self.b, 1.0 if self.nonlin else 0.0]

    @classmethod
    def init(cls, nin: int, nonlin: bool = True) -> "Neuron":
        w = [random.uniform(-1, 1) for _ in range(nin)]

        return Neuron(w=w, b=0.0, nonlin=nonlin)

    @classmethod
    def from_parameters(cls, params: list[Scalar]) -> "Neuron":
        w = params[:-2]
        b = params[-2]
        nonlin = bool(params[-1])

        return cls(w=w, b=b, nonlin=nonlin)


@dataclass(frozen=True)
class Layer:
    neurons: list[Neuron]

    def __call__(self, x: list[Scalar]) -> Scalar | list[Scalar]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Scalar]:
        return [p for n in self.neurons for p in n.parameters()]

    @classmethod
    def init(cls, nin: int, nout: int, **kwargs) -> "Layer":
        return Layer([Neuron.init(nin, **kwargs) for _ in range(nout)])

    @classmethod
    def from_parameters(cls, params: list[Scalar], nin: int, nout: int) -> "Layer":
        width = nin + 2
        neurons = [
            Neuron.from_parameters(params[i * width : (i + 1) * width])
            for i in range(nout)
        ]
        return cls(neurons)


@dataclass(frozen=True)
class MLP:
    layers: list[Layer]

    def __call__(self, x: list[Scalar]) -> list[Scalar]:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Scalar]:
        return [p for l in self.layers for p in l.parameters()]

    @classmethod
    def init(cls, nin: int, nouts: list[int]) -> "MLP":
        sz = [nin] + nouts
        layers = [
            Layer.init(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]
        return cls(layers)

    @classmethod
    def from_params(cls, params: list[Scalar], nin: int, nouts: list[int]) -> "MLP":
        pass
