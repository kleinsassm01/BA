import logging

import netket as nk

from .autocorr import analyze_vstate_energy
from .config import AutocorrConfig, ModelConfig, TrainingConfig
from .exact_solver import ExactIsingSolver
from .models import PointHistory, TrainingResult
from .problem_factory import IsingProblemFactory


class TFIMTrainer:
    """
    Trainer for the 1D TFIM with NQS. Extended from the original so that:

      * The per-step integrated autocorrelation time tau_corr reported by
        NetKet's vstate.expect(H).tau_corr is recorded at every log step.
        This is a cheap in-training diagnostic -- it tells us whether the
        MC chain is well-mixed at a given J.

      * <m^4> is recorded so the Binder cumulant U_4 can be computed.

      * After training converges, an optional long dedicated MC chain is run
        on the final variational state and the integrated autocorrelation
        time of the local-energy series is estimated via Sokal windowing.
        This is the "proper" MCMC diagnostic; it shows critical slowing
        down at the quantum critical point.
    """

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

    def train_point(self, J: float) -> TrainingResult:
        N = self.model_cfg.N
        h = self.model_cfg.h
        cfg = self.train_cfg

        hamiltonian, hilbert, _ = self.factory.build_hamiltonian(N, J, h)
        m2_op, n2_op, m4_op = self.factory.build_observables(hilbert, N)

        model = nk.models.RBM(alpha=cfg.alpha)
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=cfg.n_chains)

        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=cfg.n_samples,
            n_discard_per_chain=cfg.n_discard_per_chain,
        )

        optimizer = nk.optimizer.Sgd(learning_rate=cfg.lr)
        sr = nk.optimizer.SR(diag_shift=cfg.sr_diag_shift)

        gs = nk.VMC(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            preconditioner=sr,
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
                m4_val = vstate.expect(m4_op)

                history.iters.append(step)
                history.energy.append(float(e.mean.real / N))
                history.e_var.append(float(e.variance.real / N))
                history.m2.append(float(m2_val.mean.real))
                history.n2.append(float(n2_val.mean.real))
                history.m4.append(float(m4_val.mean.real))
                # NetKet's Stats object exposes tau_corr directly. Cast to
                # float and guard against None for very early steps.
                tau = getattr(e, "tau_corr", None)
                history.tau_corr.append(
                    float(tau) if (tau is not None and tau > 0) else float("nan")
                )

        # --- post-training dedicated autocorrelation analysis ----------------
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
                # An ACF failure must never take down the whole sweep.
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
            "Finished J=%.3f | E/N(NQS)=%.6f | E/N(exact)=%.6f | err=%.3f%% | m2=%.4f | n2=%.4f%s",
            result.J,
            result.e_final,
            result.e_exact_finite,
            result.rel_error_pct,
            result.m2_final,
            result.n2_final,
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
