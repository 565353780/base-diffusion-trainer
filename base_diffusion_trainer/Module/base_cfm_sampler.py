import torch

from torch import nn
from tqdm import tqdm
from typing import Iterable, Optional, Sequence, Union

from base_diffusion_trainer.Module.stacked_random_generator import StackedRandomGenerator


class BaseCFMSampler(object):
    """Reusable CFM sampling primitives shared by trainer and inference code.

    Conventions enforced by this base class:

    - Time axis is ``t_cfm`` increasing from 0 (pure noise) to 1 (data).
    - Linear flow path: ``x(t) = (1 - t) * x_noise + t * x_target``.
    - Target / sampling velocity is ``v_cfm = x_target - x_noise``, so the ODE
      reads ``dx/dt_cfm = v_cfm`` and a discrete Euler step is
      ``x <- x + dt_cfm * v_cfm`` with ``dt_cfm > 0``.

    Subclasses that wrap legacy networks whose timestep embedding decreases
    from 1 to 0 and whose raw output is ``-v_cfm`` should perform the two
    conversions inside :meth:`predictVelocity` (one place, no per-loop sign
    bookkeeping).
    """

    def __init__(
        self,
        time_eps: float = 1e-5,
        time_logit_mean: float = 0.0,
        time_logit_std: float = 1.0,
        time_shift_mu: float = 1.15,
        ode_atol: float = 1e-4,
        ode_rtol: float = 1e-4,
        ode_method: str = "dopri5",
    ) -> None:
        self.time_eps = time_eps
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.time_shift_mu = time_shift_mu
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.ode_method = ode_method
        return

    # ------------------------------------------------------------------
    # Time / path utilities (shared with training code).
    # ------------------------------------------------------------------

    def sampleTime(
        self,
        batch_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample CFM training time ``t_cfm`` in ``[time_eps, 1 - time_eps]``.

        Uses the logit-normal distribution and the Flux-style time shift
        ``time_shift_mu``, then flips to the CFM convention so that larger
        values correspond to states closer to data.
        """
        logit_t = torch.randn(batch_size, device=device, dtype=dtype)
        logit_t = logit_t * self.time_logit_std + self.time_logit_mean
        t = torch.sigmoid(logit_t)
        t = t.clamp(self.time_eps, 1.0 - self.time_eps)

        shifted_t = torch.exp(torch.tensor(self.time_shift_mu, device=device, dtype=dtype))
        t = shifted_t / (shifted_t + (1.0 / t - 1.0))
        inv_t = 1.0 - t
        return inv_t.clamp(self.time_eps, 1.0 - self.time_eps)

    def expandTimeLike(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        return t

    def sampleFlowPath(
        self,
        x_noise: torch.Tensor,
        x_target: torch.Tensor,
        t: torch.Tensor,
    ):
        """Build ``(x_noisy_target, v_cfm)`` for a batch of CFM time samples."""
        t = self.expandTimeLike(t, x_noise)
        x_noisy_target = (1.0 - t) * x_noise + t * x_target
        v = x_target - x_noise
        return x_noisy_target, v

    # ------------------------------------------------------------------
    # Hooks intended for subclass override.
    # ------------------------------------------------------------------

    def buildInitialNoise(
        self,
        data_shape: Sequence[int],
        device: Union[str, torch.device],
        dtype: torch.dtype,
        seed: Optional[int] = None,
        sample_num: Optional[int] = None,
    ) -> torch.Tensor:
        """Default noise initializer used by :meth:`sampleData`.

        Subclasses may override to plug in custom noise layouts or external
        random sources (e.g. Flux's ``get_noise``).
        """
        if sample_num is None:
            sample_num = data_shape[0]

        base_seed = 0 if seed is None else int(seed)
        batch_seeds = torch.arange(sample_num) + base_seed
        rnd = StackedRandomGenerator(device, batch_seeds)
        return rnd.randn(list(data_shape), device=device, dtype=dtype)

    def buildTimeSchedule(
        self,
        num_steps: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Default ``t_cfm`` schedule for the discrete Euler/Heun sampler.

        Returns ``num_steps + 1`` strictly increasing values in
        ``[time_eps, 1 - time_eps]``. Subclasses can override to inject
        bespoke schedules (e.g. shifted Flux timesteps converted to ``t_cfm``).
        """
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        return ts.clamp(self.time_eps, 1.0 - self.time_eps)

    def predictVelocity(
        self,
        model: nn.Module,
        x_noisy_target: torch.Tensor,
        t_cfm: torch.Tensor,
        model_input_dict: dict,
        step_idx: int,
    ) -> torch.Tensor:
        """Return the CFM velocity ``v_cfm`` evaluated at ``(x, t_cfm)``.

        The default implementation matches the convention used by the CFM
        trainer / CombineModel pair: stuff ``t_cfm`` and ``x_noisy_target``
        into the model input dict, run the model, post-process via
        :meth:`postProcessData`, and read ``result_dict['v']``.

        Legacy subclasses (e.g. the Flux ``Detector``) should override this
        and perform any ``t``/``v`` direction conversions in one place.
        """
        input_dict = dict(model_input_dict)
        input_dict["t"] = t_cfm
        input_dict["x_noisy_target"] = x_noisy_target
        result_dict = model(input_dict)
        result_dict = self.postProcessData(input_dict, result_dict, False)
        return result_dict["v"]

    def postProcessData(
        self,
        data_dict: dict,
        result_dict: dict,
        is_training: bool = True,
    ) -> dict:
        """Hook mirroring :meth:`BaseTrainer.postProcessData`.

        Provided so :meth:`predictVelocity` works without inheriting
        ``BaseTrainer``; trainers will override / shadow this with their own
        implementation as today.
        """
        return result_dict

    def postProcessSampleStep(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t_cfm_curr: torch.Tensor,
        t_cfm_next: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Hook called after every Euler/Heun integrator step.

        Receives the post-step latent ``x`` and may return a modified tensor
        (e.g. for re-projection onto a manifold). Default is identity.
        """
        return x

    def makeSampleProgressBar(
        self,
        iterable: Iterable,
        total: int,
        desc: str,
    ):
        """Override to disable / customize the progress bar (e.g. on non-rank-0)."""
        return tqdm(iterable, total=total, desc=desc)

    # ------------------------------------------------------------------
    # Adaptive ODE sampler (matches the legacy ``BaseCFMTrainer.sampleData``).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sampleData(
        self,
        model: nn.Module,
        model_input_dict: dict,
        data_shape: Sequence[int],
        sample_num: int = 1,
        timestamp_num: int = 2,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Adaptive-step ODE sampler returning the final latent on CPU.

        Matches the original ``BaseCFMTrainer.sampleData`` API exactly so
        existing demos / call sites keep working after the refactor.
        """
        device = self._resolveDevice()
        dtype = self._resolveDtype()

        query_t = torch.linspace(0.0, 1.0, timestamp_num).to(device=device, dtype=dtype)

        x_init = self.buildInitialNoise(
            data_shape=data_shape,
            device=device,
            dtype=dtype,
            seed=seed,
            sample_num=sample_num,
        )

        def ode_fn(t: torch.Tensor, x_noisy_target: torch.Tensor) -> torch.Tensor:
            t = t.to(device=x_noisy_target.device, dtype=x_noisy_target.dtype)
            if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
                t = t.reshape(1).expand(x_noisy_target.shape[0])

            return self.predictVelocity(
                model=model,
                x_noisy_target=x_noisy_target,
                t_cfm=t,
                model_input_dict=model_input_dict,
                step_idx=-1,
            )

        # ``torchdiffeq`` is an optional runtime dependency; import lazily so
        # consumers that only use the discrete Euler/Heun sampler don't need
        # it installed.
        import torchdiffeq

        traj = torchdiffeq.odeint(
            ode_fn,
            x_init,
            query_t,
            atol=self.ode_atol,
            rtol=self.ode_rtol,
            method=self.ode_method,
        )

        return traj.cpu()[-1]

    # ------------------------------------------------------------------
    # Discrete Euler / Heun sampler (replaces the legacy detector loop).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sampleLatentByEulerOrHeun(
        self,
        model: nn.Module,
        x_init: torch.Tensor,
        model_input_dict: Optional[dict] = None,
        num_steps: int = 25,
        solver: str = "heun",
        time_schedule: Optional[torch.Tensor] = None,
        progress_desc: str = "Sampling latent",
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Discrete CFM integrator on the increasing ``t_cfm`` axis.

        Each step performs ``x <- x + dt_cfm * v_cfm`` (and the standard
        Heun trapezoid correction when ``solver='heun'``). All direction
        bookkeeping for legacy networks should live inside
        :meth:`predictVelocity`, not here.

        Args:
            model: Network passed to :meth:`predictVelocity`. Subclasses may
                ignore it if they capture the network elsewhere.
            x_init: Initial noise latent (``t_cfm = time_schedule[0]``).
            model_input_dict: Optional shared inputs forwarded to
                :meth:`predictVelocity` (e.g. ``guidance``, condition tokens).
            num_steps: Number of integration intervals when
                ``time_schedule`` is not provided.
            solver: ``'euler'`` or ``'heun'``.
            time_schedule: Optional pre-computed ``t_cfm`` schedule. Must be
                strictly increasing in ``[0, 1]`` with length
                ``num_steps + 1``. If omitted, :meth:`buildTimeSchedule` is
                used.
            progress_desc: tqdm description.
            show_progress: When False, skip tqdm (e.g. non-logger ranks).
        """
        if solver not in ("euler", "heun"):
            raise ValueError(f"Unsupported CFM solver: {solver!r}")

        if model_input_dict is None:
            model_input_dict = {}

        device = x_init.device
        dtype = x_init.dtype

        if time_schedule is None:
            time_schedule = self.buildTimeSchedule(
                num_steps=num_steps, device=device, dtype=dtype,
            )
        else:
            time_schedule = time_schedule.to(device=device, dtype=dtype)

        if time_schedule.ndim != 1 or time_schedule.numel() < 2:
            raise ValueError(
                "time_schedule must be a 1-D tensor with at least 2 entries; "
                f"got shape {tuple(time_schedule.shape)}"
            )

        x = x_init
        num_iters = time_schedule.numel() - 1

        pairs = list(zip(time_schedule[:-1].tolist(), time_schedule[1:].tolist()))
        if show_progress:
            iterator = self.makeSampleProgressBar(
                enumerate(pairs), total=num_iters, desc=progress_desc,
            )
        else:
            iterator = enumerate(pairs)

        use_heun = solver == "heun"

        for i, (t_curr_val, t_next_val) in iterator:
            dt = t_next_val - t_curr_val
            is_last = (i == num_iters - 1)

            t_curr = self._fillTimeTensor(t_curr_val, x.shape[0], device, dtype)
            v1 = self.predictVelocity(
                model=model,
                x_noisy_target=x,
                t_cfm=t_curr,
                model_input_dict=model_input_dict,
                step_idx=i,
            )

            if use_heun and not is_last:
                x_euler = x + dt * v1
                t_next = self._fillTimeTensor(t_next_val, x.shape[0], device, dtype)
                v2 = self.predictVelocity(
                    model=model,
                    x_noisy_target=x_euler,
                    t_cfm=t_next,
                    model_input_dict=model_input_dict,
                    step_idx=i,
                )
                x_new = x + dt * 0.5 * (v1 + v2)
                del x_euler, v2
            else:
                x_new = x + dt * v1

            t_curr_full = self._fillTimeTensor(t_curr_val, x.shape[0], device, dtype)
            t_next_full = self._fillTimeTensor(t_next_val, x.shape[0], device, dtype)
            x = self.postProcessSampleStep(
                x=x_new,
                v=v1,
                t_cfm_curr=t_curr_full,
                t_cfm_next=t_next_full,
                step_idx=i,
            )
            del v1

        return x

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _resolveDevice(self) -> Union[str, torch.device]:
        device = getattr(self, "device", None)
        if device is None:
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def _resolveDtype(self) -> torch.dtype:
        dtype = getattr(self, "dtype", None)
        if dtype is None:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return dtype

    @staticmethod
    def _fillTimeTensor(
        value: float,
        batch_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.full((batch_size,), float(value), device=device, dtype=dtype)
