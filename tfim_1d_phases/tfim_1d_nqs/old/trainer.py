import logging

import netket as nk

from .config import ModelConfig, TrainingConfig
from .exact_solver import ExactIsingSolver
from .models import PointHistory, TrainingResult
from .problem_factory import IsingProblemFactory


class TFIMTrainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        logger: logging.Logger,
        exact_solver: ExactIsingSolver | None = None,
        factory: IsingProblemFactory | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.logger = logger
        self.exact_solver = exact_solver or ExactIsingSolver()
        self.factory = factory or IsingProblemFactory()

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
        sr = nk.optimizer.SR(diag_shift=cfg.sr_diag_shift)

        gs = nk.VMC(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            preconditioner=sr,
            variational_state=vstate,
        )

        history = PointHistory(iters=[], energy=[], e_var=[], m2=[], n2=[])

        e_exact_finite = self.exact_solver.energy_finite(N, J, h)
        e_exact_thermo = self.exact_solver.energy_thermodynamic(J, h)

        for step in range(cfg.n_iter):
            gs.advance()

            if step % cfg.log_every == 0 or step == cfg.n_iter - 1:
                e = vstate.expect(hamiltonian)
                m2_val = vstate.expect(m2_op)
                n2_val = vstate.expect(n2_op)

                history.iters.append(step)
                history.energy.append(float(e.mean.real / N))
                history.e_var.append(float(e.variance.real / N))
                history.m2.append(float(m2_val.mean.real))
                history.n2.append(float(n2_val.mean.real))

        result = TrainingResult(
            J=float(J),
            h=float(h),
            e_exact_finite=float(e_exact_finite),
            e_exact_thermo=float(e_exact_thermo),
            history=history,
        )

        self.logger.info(
            "Finished J=%.3f | E/N(NQS)=%.6f | E/N(exact)=%.6f | err=%.3f%% | m2=%.4f | n2=%.4f",
            result.J,
            result.e_final,
            result.e_exact_finite,
            result.rel_error_pct,
            result.m2_final,
            result.n2_final,
        )
        return result

    def scan(self, J_values) -> list[TrainingResult]:
        results: list[TrainingResult] = []

        for idx, J in enumerate(J_values, start=1):
            self.logger.info(
                "Training point %d/%d | J=%.3f | h=%.3f",
                idx, len(J_values), J, self.model_cfg.h
            )
            results.append(self.train_point(float(J)))

        return results