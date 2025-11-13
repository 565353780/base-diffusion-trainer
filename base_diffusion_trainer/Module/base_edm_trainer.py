import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from base_diffusion_trainer.Loss.edm import EDMLoss
from base_diffusion_trainer.Method.sample import edm_sampler
from base_diffusion_trainer.Module.stacked_random_generator import StackedRandomGenerator


class BaseEDMTrainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
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
        use_amp: bool = False,
        quick_test: bool = False,
    ) -> None:
        self.loss_func = EDMLoss()

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def preProcessDiffusionData(
        self, data_dict: dict, data_name: str, is_training: bool = False
    ) -> dict:
        data = data_dict[data_name]

        noise, sigma, weight = self.loss_func(data, not is_training)

        data_dict["noise"] = noise
        data_dict["sigma"] = sigma
        data_dict["weight"] = weight

        return data_dict

    def getDiffusionLoss(self, data_dict: dict, data_name: str, result_dict: dict) -> torch.Tensor:
        inputs = data_dict[data_name]
        D_yn = result_dict["D_x"]
        weight = data_dict["weight"]

        loss_diffusion = weight * ((D_yn - inputs) ** 2)

        loss_diffusion = loss_diffusion.mean()

        return loss_diffusion

    @torch.no_grad()
    def sampleData(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        data_shape: list,
        sample_num: int,
        timestamp_num: int = 2,
    ) -> torch.Tensor:
        timestamp_num = 18

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        latents = rnd.randn(data_shape, device=self.device)

        sampled_array = edm_sampler(
            model,
            latents,
            condition,
            randn_like=rnd.randn_like,
            num_steps=timestamp_num,
        )[-1]

        return sampled_array
