import numpy as np
from scipy import integrate
from scipy.special import ellipe


class ExactIsingSolver:
    @staticmethod
    def energy_finite(N: int, J: float, h: float) -> float:
        k_r = 2.0 * np.pi * np.arange(N) / N
        eps_r = 2.0 * np.sqrt(J**2 + h**2 + 2.0 * J * h * np.cos(k_r))
        e_r = -0.5 * np.sum(eps_r)

        k_ns = 2.0 * np.pi * (np.arange(N) + 0.5) / N
        eps_ns = 2.0 * np.sqrt(J**2 + h**2 + 2.0 * J * h * np.cos(k_ns))
        e_ns = -0.5 * np.sum(eps_ns)

        return min(e_r, e_ns) / N

    @staticmethod
    def energy_thermodynamic(J: float, h: float) -> float:
        abs_j = abs(J)
        if abs_j == 0 and h == 0:
            return 0.0

        a = max(abs_j, h)
        b = min(abs_j, h)
        if a == 0:
            return 0.0

        m = (b / a) ** 2
        return -(2.0 / np.pi) * a * float(ellipe(m))

    @staticmethod
    def energy_numerical_integral(J: float, h: float) -> float:
        def integrand(k: float) -> float:
            return np.sqrt(J**2 + h**2 + 2.0 * J * h * np.cos(k))

        result, _ = integrate.quad(integrand, 0, 2 * np.pi)
        return -result / (2.0 * np.pi)