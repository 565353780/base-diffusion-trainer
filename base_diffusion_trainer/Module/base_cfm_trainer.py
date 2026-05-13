import torch

from torch import nn
from typing import Callable, Optional, Union
from torch.distributed.fsdp import MixedPrecisionPolicy

from base_trainer.Module.base_trainer import BaseTrainer
from base_trainer.Method.fsdp import default_fsdp_shard_fn

from base_diffusion_trainer.Module.base_cfm_sampler import BaseCFMSampler


class BaseCFMTrainer(BaseTrainer, BaseCFMSampler):
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
        is_balanced_sample: bool = False,
        time_eps: float = 1e-5,
        time_logit_mean: float = 0.0,
        time_logit_std: float = 1.0,
        time_shift_mu: float = 1.15,
        ode_atol: float = 1e-4,
        ode_rtol: float = 1e-4,
        ode_method: str = "dopri5",
    ) -> None:
        BaseCFMSampler.__init__(
            self,
            time_eps=time_eps,
            time_logit_mean=time_logit_mean,
            time_logit_std=time_logit_std,
            time_shift_mu=time_shift_mu,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
            ode_method=ode_method,
        )

        self.diffusion_loss_fn = nn.MSELoss()

        BaseTrainer.__init__(
            self,
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
            is_balanced_sample=is_balanced_sample,
        )
        return

    def preProcessDiffusionData(
        self,
        data_dict: dict,
    ) -> dict:
        x_target = data_dict["x_target"]
        x_noise = torch.randn_like(x_target)

        t = self.sampleTime(
            x_target.shape[0],
            x_target.device,
            x_target.dtype,
        )
        x_noisy_target, v = self.sampleFlowPath(x_noise, x_target, t)

        data_dict["x_noise"] = x_noise
        data_dict["x_target"] = x_target
        data_dict["t"] = t
        data_dict["x_noisy_target"] = x_noisy_target
        data_dict["v"] = v

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        loss_diffusion = result_dict['loss_diffusion']
        loss_dict = {
            'Loss': loss_diffusion,
            'loss_diffusion': loss_diffusion,
        }
        return loss_dict

    def getDiffusionLoss(
        self,
        data_dict: dict,
        result_dict: dict,
    ) -> torch.Tensor:
        target_velocity = data_dict["v"]
        prediction_velocity = result_dict["v"]
        loss_diffusion = self.diffusion_loss_fn(
            prediction_velocity.float(),
            target_velocity.float(),
        )
        return loss_diffusion
