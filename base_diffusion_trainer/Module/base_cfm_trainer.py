import torch
import torchdiffeq
from torch import nn
from typing import Union

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from base_trainer.Module.base_trainer import BaseTrainer

from base_diffusion_trainer.Module.batch_ot_cfm import (
    BatchExactOptimalTransportConditionalFlowMatcher,
)
from base_diffusion_trainer.Module.stacked_random_generator import StackedRandomGenerator


class BaseCFMTrainer(BaseTrainer):
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
        fm_id = 3
        if fm_id == 1:
            self.FM = ConditionalFlowMatcher(sigma=0.0)
        elif fm_id == 2:
            # FIXME: this module will mismatch the condition and shapes! do not use it for now!
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        elif fm_id == 3:
            self.FM = AffineProbPath(scheduler=CondOTScheduler())
        elif fm_id == 4:
            # TODO: this is the best one, but too slow for large data, need to speed up in the future
            self.FM = BatchExactOptimalTransportConditionalFlowMatcher(
                sigma=0.0, target_dim=[0, 1, 2]
            )

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
        self, data_dict: dict, data_name: str, is_training: bool = False,
    ) -> dict:
        data = data_dict[data_name]

        init_data = torch.randn_like(data)

        if isinstance(self.FM, ConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_data, data
            )
        elif isinstance(self.FM, ExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_data, data
            )
        elif isinstance(self.FM, BatchExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_data, data
            )
        elif isinstance(self.FM, AffineProbPath):
            t = torch.rand(data.shape[0]).to(
                data.device, dtype=data.dtype
            )
            t = torch.pow(t, 0.5)
            path_sample = self.FM.sample(t=t, x_0=init_data, x_1=data)
            t = path_sample.t
            xt = path_sample.x_t
            ut = path_sample.dx_t
        else:
            print("[ERROR][BaseCFMTrainer::preProcessDiffusionData]")
            print("\t FM not valid!")
            exit()

        data_dict["ut"] = ut
        data_dict["t"] = t
        data_dict["xt"] = xt

        return data_dict

    def getDiffusionLoss(self, data_dict: dict, result_dict: dict) -> torch.Tensor:
        ut = data_dict["ut"]
        vt = result_dict["vt"]

        loss_diffusion = torch.pow(vt - ut, 2).mean()

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
        query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 0.5)

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        x_init = rnd.randn(data_shape, device=self.device)

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
