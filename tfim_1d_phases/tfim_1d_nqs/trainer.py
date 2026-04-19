import logging

import netket as nk
import numpy as np

from .autocorr import analyze_vstate_energy
from .config import AutocorrConfig, ModelConfig, TrainingConfig
from .exact_solver import ExactIsingSolver
from .models import PointHistory, TrainingResult
from .problem_factory import IsingProblemFactory


class TFIMTrainer:

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        autocorr_cfg: AutocorrConfig | None = None,
        exact_solver: ExactIsingSolver | None = None,
        factory: IsingProblemFactory | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.logger = logger
        self.autocorr_cfg = autocorr_cfg or AutocorrConfig(enabled=False)
        self.exact_solver = exact_solver or ExactIsingSolver()
        self.factory = factory or IsingProblemFactory()

    @staticmethod
    def _measure_m4_from_samples(vstate, N: int) -> float:
        vstate.sample()
        samples = np.asarray(vstate.samples)
        # NetKet shape: (n_chains, n_per_chain, N).
        if samples.ndim == 3:
            samples = samples.reshape(-1, samples.shape[-1])
            
        # spins must be in {-1, +1}.
        assert samples.shape[-1] == N, (
            f"sample last-dim {samples.shape[-1]} != N={N}"
        )
        m_per_sample = samples.mean(axis=-1)
        if m_per_sample.min() >= -0.01 and m_per_sample.max() <= 1.01 \
                and samples.min() >= -0.01:
            m_per_sample = 2.0 * m_per_sample - 1.0
        return float(np.mean(m_per_sample ** 4))

    def train_point(self, J: float) -> TrainingResult:
        N = self.model_cfg.N
        h = self.model_cfg.h
        cfg = self.train_cfg

        hamiltonian, hilbert, _ = self.factory.build_hamiltonian(N, J, h)
        m2_op, n2_op = self.factory.build_observables(hilbert, N)

        model = nk.models.RBM(alpha=cfg.alpha)
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=cfg.n_chains)

        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=cfg.n_samples,
            n_discard_per_chain=cfg.n_discard_per_chain,
        )

        optimizer = nk.optimizer.Sgd(learning_rate=cfg.lr)

        gs = nk.driver.VMC_SR(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            diag_shift=cfg.sr_diag_shift,
            variational_state=vstate,
        )

        history = PointHistory(
            iters=[], energy=[], e_var=[], m2=[], n2=[],
            tau_corr=[], m4=[],
        )

        e_exact_finite = self.exact_solver.energy_finite(N, J, h)
        e_exact_thermo = self.exact_solver.energy_thermodynamic(J, h)

        for step in range(cfg.n_iter):
            gs.advance()

            if step % cfg.log_every == 0 or step == cfg.n_iter - 1:
                e = vstate.expect(hamiltonian)
                m2_val = vstate.expect(m2_op)
                n2_val = vstate.expect(n2_op)
                
                m4_sample = self._measure_m4_from_samples(vstate, N)

                history.iters.append(step)
                history.energy.append(float(e.mean.real / N))
                history.e_var.append(float(e.variance.real / N))
                history.m2.append(float(m2_val.mean.real))
                history.n2.append(float(n2_val.mean.real))
                history.m4.append(m4_sample)

                tau = getattr(e, "tau_corr", None)
                history.tau_corr.append(
                    float(tau) if (tau is not None and tau > 0) else float("nan")
                )

        autocorr_result = None
        if self.autocorr_cfg.enabled:
            try:
                autocorr_result = analyze_vstate_energy(
                    vstate=vstate,
                    hamiltonian=hamiltonian,
                    n_samples=self.autocorr_cfg.n_samples,
                    n_chains=self.autocorr_cfg.n_chains,
                    n_discard=self.autocorr_cfg.n_discard,
                    max_lag=self.autocorr_cfg.max_lag,
                    sokal_c=self.autocorr_cfg.sokal_c,
                    logger=self.logger,
                )
            except Exception as exc:
                self.logger.warning(
                    "  autocorr analysis failed at J=%.3f: %s", J, exc
                )
                autocorr_result = None

        result = TrainingResult(
            J=float(J),
            h=float(h),
            N=int(N),
            e_exact_finite=float(e_exact_finite),
            e_exact_thermo=float(e_exact_thermo),
            history=history,
            autocorr=autocorr_result,
        )

        self.logger.info(
            "Finished J=%.3f | E/N(NQS)=%.6f | E/N(exact)=%.6f | err=%.3f%% | "
            "m2=%.4f | n2=%.4f | m4=%.4f%s",
            result.J,
            result.e_final,
            result.e_exact_finite,
            result.rel_error_pct,
            result.m2_final,
            result.n2_final,
            result.m4_final,
            f" | tau_int={autocorr_result.tau_int:.2f}" if autocorr_result is not None else "",
        )
        return result

    def scan(self, J_values) -> list[TrainingResult]:
        results: list[TrainingResult] = []
        for idx, J in enumerate(J_values, start=1):
            self.logger.info(
                "Training point %d/%d | N=%d | J=%.3f | h=%.3f",
                idx, len(J_values), self.model_cfg.N, J, self.model_cfg.h
            )
            results.append(self.train_point(float(J)))
        return results
