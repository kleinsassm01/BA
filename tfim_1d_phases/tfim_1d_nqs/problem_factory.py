import netket as nk


class IsingProblemFactory:
    # netket convention: H = -h * sum_i sigma^x_i  +  J * sum_<ij> sigma^z_i sigma^z_j

    @staticmethod
    def build_hamiltonian(N: int, J: float, h: float):
        graph = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=h, J=J)
        return hamiltonian, hilbert, graph

    @staticmethod
    def build_observables(hilbert, N: int):
        # <m^4> is NOT built as an operator here cost would be O(N^4)
        # terms and was the bottleneck at large N. Instead the trainer
        # estimates <m^4> directly from the MC samples (see trainer.py).
        sigma_z = nk.operator.spin.sigmaz
        Sz = sum(sigma_z(hilbert, i) for i in range(N))
        m2_op = (Sz @ Sz) * (1.0 / (N * N))

        Nz = sum(((-1) ** i) * sigma_z(hilbert, i) for i in range(N))
        n2_op = (Nz @ Nz) * (1.0 / (N * N))

        return m2_op, n2_op
