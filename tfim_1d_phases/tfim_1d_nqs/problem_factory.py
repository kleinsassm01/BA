import netket as nk


class IsingProblemFactory:
    @staticmethod
    def build_hamiltonian(N: int, J: float, h: float):
        graph = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=h, J=J)
        return hamiltonian, hilbert, graph

    @staticmethod
    def build_observables(hilbert, N: int):
        sigma_z = nk.operator.spin.sigmaz

        m2_op = sum(
            sigma_z(hilbert, i) * sigma_z(hilbert, j)
            for i in range(N) for j in range(N)
        ) / (N * N)

        n2_op = sum(
            ((-1) ** (i + j)) * sigma_z(hilbert, i) * sigma_z(hilbert, j)
            for i in range(N) for j in range(N)
        ) / (N * N)

        return m2_op, n2_op