from __future__ import annotations

import netket as nk
from netket.operator.spin import sigmax, sigmaz


def make_tfim(N: int = 20, J: float = 1.0, h: float = 1.0, pbc: bool = True):
    hi = nk.hilbert.Spin(s=0.5, N=N)
    graph = nk.graph.Chain(length=N, pbc=pbc)

    H = sum(-J * sigmaz(hi, i) @ sigmaz(hi, j) for i, j in graph.edges())
    H += sum(-h * sigmax(hi, i) for i in range(N))
    return hi, graph, H
