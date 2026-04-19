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
        """
        Standard order-parameter observables:

            m^2 = (1/N^2) sum_{i,j} sigma^z_i sigma^z_j           (ferro)
            n^2 = (1/N^2) sum_{i,j} (-1)^(i+j) sigma^z_i sigma^z_j (antiferro)

        Plus the fourth moment of the magnetization:

            m^4 = (1/N^4) sum_{i,j,k,l} sigma^z_i sigma^z_j sigma^z_k sigma^z_l

        <m^4> is needed for the Binder cumulant U_4 = 1 - <m^4> / (3 <m^2>^2),
        which is the standard diagnostic for a second-order phase transition.

        Note: the m^4 operator is O(N^4) terms, so construction is expensive
        for large N. For N ~ 80 this is still fine (~4e7 two-body terms once
        NetKet simplifies), but we build it from Mx = sum_i sigma^z_i / N so
        that we rely on polynomial composition rather than a literal
        quadruple sum.
        """
        sigma_z = nk.operator.spin.sigmaz

        # Total magnetization operator (unscaled): Sz = sum_i sigma^z_i
        Sz = sum(sigma_z(hilbert, i) for i in range(N))

        # <m^2> with m = Sz / N  ->  (1/N^2) * Sz^2
        m2_op = (Sz @ Sz) * (1.0 / (N * N))

        # <m^4> = (1/N^4) * Sz^4
        Sz2 = Sz @ Sz
        m4_op = (Sz2 @ Sz2) * (1.0 / (N ** 4))

        # Staggered magnetization Nz = sum_i (-1)^i sigma^z_i
        Nz = sum(((-1) ** i) * sigma_z(hilbert, i) for i in range(N))
        n2_op = (Nz @ Nz) * (1.0 / (N * N))

        return m2_op, n2_op, m4_op
