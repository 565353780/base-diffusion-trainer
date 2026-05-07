import torch
import torchdiffeq
from torch import nn
from typing import Callable, Optional, Union
from torch.distributed.fsdp import MixedPrecisionPolicy

from base_trainer.Module.base_trainer import BaseTrainer
from base_trainer.Method.fsdp import default_fsdp_shard_fn

from base_diffusion_trainer.Module.stacked_random_generator import StackedRandomGenerator


class BaseCFMTrainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        prefetch_factor: int = 2,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        quick_test: bool = False,
        save_checkpoint_freq: int = -1,
        fsdp_shard_fn: Optional[Callable] = default_fsdp_shard_fn,
        compile_fn: Optional[Callable] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        load_model_fn: Optional[Callable] = None,
        save_model_fn: Optional[Callable] = None,
        time_eps: float = 1e-5,
        time_logit_mean: float = 0.0,
        time_logit_std: float = 1.0,
        time_shift_mu: float = 1.15,
        path_direction: str = "data_to_noise",
        diffusion_loss_clamp: Optional[float] = None,
    ) -> None:
        self.time_eps = time_eps
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.time_shift_mu = time_shift_mu
        self.path_direction = path_direction
        self.diffusion_loss_clamp = diffusion_loss_clamp

        super().__init__(
            batch_size=batch_size,
            accum_iter=accum_iter,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            model_file_path=model_file_path,
            weights_only=weights_only,
            warm_step_num=warm_step_num,
            finetune_step_num=finetune_step_num,
            lr=lr,
            lr_batch_size=lr_batch_size,
            ema_start_step=ema_start_step,
            ema_decay_init=ema_decay_init,
            ema_decay=ema_decay,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            best_model_metric_name=best_model_metric_name,
            is_metric_lower_better=is_metric_lower_better,
            sample_results_freq=sample_results_freq,
            quick_test=quick_test,
            save_checkpoint_freq=save_checkpoint_freq,
            fsdp_shard_fn=fsdp_shard_fn,
            compile_fn=compile_fn,
            mp_policy=mp_policy,
            load_model_fn=load_model_fn,
            save_model_fn=save_model_fn,
        )
        return

    def sampleTime(self, batch_size: int, device, dtype) -> torch.Tensor:
        logit_t = torch.randn(batch_size, device=device, dtype=dtype)
        logit_t = logit_t * self.time_logit_std + self.time_logit_mean
        t = torch.sigmoid(logit_t)
        t = t.clamp(self.time_eps, 1.0 - self.time_eps)

        shifted_t = torch.exp(torch.tensor(self.time_shift_mu, device=device, dtype=dtype))
        t = shifted_t / (shifted_t + (1.0 / t - 1.0))
        return t.clamp(self.time_eps, 1.0 - self.time_eps)

    def expandTimeLike(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        return t

    def sampleFlowPath(
        self,
        path_start: torch.Tensor,
        path_end: torch.Tensor,
        t: torch.Tensor,
    ):
        t = self.expandTimeLike(t, path_start)
        x_t = (1.0 - t) * path_start + t * path_end
        v_t = path_end - path_start
        return x_t, v_t

    def _getPathEndpoints(self, clean_data: torch.Tensor, noise_data: torch.Tensor):
        if self.path_direction == "data_to_noise":
            return clean_data, noise_data
        if self.path_direction == "noise_to_data":
            return noise_data, clean_data
        raise ValueError(f"Unknown path_direction: {self.path_direction}")

    def preProcessDiffusionData(
        self,
        data_dict: dict,
        data_name: str,
        is_training: bool = False,
        noise_data_name: str = "x_0",
        clean_data_name: str = "x_1",
        time_name: str = "t",
        output_data_name: str = "x_t",
        target_velocity_name: str = "v_t",
        write_legacy_names: bool = True,
    ) -> dict:
        clean_data = data_dict[data_name]
        noise_data = torch.randn_like(clean_data)
        path_start, path_end = self._getPathEndpoints(clean_data, noise_data)

        t = self.sampleTime(
            clean_data.shape[0],
            clean_data.device,
            clean_data.dtype,
        )
        xt, ut = self.sampleFlowPath(path_start, path_end, t)

        data_dict[noise_data_name] = noise_data
        data_dict[clean_data_name] = clean_data
        data_dict[time_name] = t
        data_dict[output_data_name] = xt
        data_dict[target_velocity_name] = ut

        if write_legacy_names:
            data_dict["xt"] = xt
            data_dict["ut"] = ut

        return data_dict

    def getDiffusionLoss(
        self,
        data_dict: dict,
        result_dict: dict,
        target_velocity_name: str = "v_t",
        prediction_velocity_name: str = "v_t",
    ) -> torch.Tensor:
        if target_velocity_name not in data_dict and target_velocity_name == "v_t":
            target_velocity_name = "ut"
        if prediction_velocity_name not in result_dict and prediction_velocity_name == "v_t":
            prediction_velocity_name = "vt"

        target_velocity = data_dict[target_velocity_name]
        prediction_velocity = result_dict[prediction_velocity_name]

        loss_diffusion = torch.pow(
            prediction_velocity.float() - target_velocity.float(),
            2,
        ).mean()

        if self.diffusion_loss_clamp is not None:
            loss_diffusion = loss_diffusion.clamp(max=self.diffusion_loss_clamp)

        return loss_diffusion

    @torch.no_grad()
    def sampleData(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        data_shape: list,
        sample_num: int = 1,
        timestamp_num: int = 2,
    ) -> torch.Tensor:
        if self.path_direction == "data_to_noise":
            query_t = torch.linspace(1, 0, timestamp_num).to(self.device)
        elif self.path_direction == "noise_to_data":
            query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        else:
            raise ValueError(f"Unknown path_direction: {self.path_direction}")

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        x_init = rnd.randn(data_shape, device=self.device, dtype=self.dtype)

        traj = torchdiffeq.odeint(
            lambda t, x: model.forwardData(x, condition, t),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        sampled_array = traj.cpu()[-1]

        return sampled_array
