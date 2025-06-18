from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn
from torch.nn.parallel import DistributedDataParallel as DDP

import sevenn._keys as KEY
from sevenn.error_recorder import ErrorRecorder

from .loss import get_loss_functions_from_config
from .optim import optim_dict, scheduler_dict


class Trainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.distributed = config[KEY.IS_DDP]

        if self.distributed:
            device = torch.device('cuda', config[KEY.LOCAL_RANK])
            dist.barrier()
            self.model = DDP(model.to(device), device_ids=[device])
            self.model.module.set_is_batch_data(True)
            self.rank = config[KEY.LOCAL_RANK]
        else:
            device = config[KEY.DEVICE]
            self.model = model.to(device)
            self.model.set_is_batch_data(True)
        self.device = device

        param = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        # This should be outside of the trainer(?)
        # list of tuples (loss_definition, weight)
        self.loss_functions = get_loss_functions_from_config(config)

    def run_one_epoch(
        self,
        loader: Iterable,
        is_train: bool = False,
        error_recorder: Optional[ErrorRecorder] = None,
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for _, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)

            output = self.model(batch)

            if is_train:
                total_loss = torch.tensor([0.0], device=self.device)
                for loss_def, w in self.loss_functions:
                    total_loss += loss_def.get_loss(output, self.model) * w

                total_loss.backward()
                self.optimizer.step()

            if error_recorder is not None:
                error_recorder.update(output)
                error_recorder.update_not_only_data(
                    self.model, self.loss_functions, output
                )

        if self.distributed and error_recorder is not None:
            self.recorder_all_reduce(error_recorder)

    def scheduler_step(self, metric: Optional[float] = None):
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            assert isinstance(metric, float)
            self.scheduler.step(metric)
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def recorder_all_reduce(self, recorder: ErrorRecorder):
        for metric in recorder.metrics:
            # metric.value._ddp_reduce(self.device)
            metric.ddp_reduce(self.device)

    def get_checkpoint_dict(self):
        if self.distributed:
            model_state_dct = self.model.module.state_dict()
        else:
            model_state_dct = self.model.state_dict()
        return {
            'model_state_dict': model_state_dct,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

    def load_state_dicts(
        self,
        model_state_dict: Dict,
        optimizer_state_dict: Dict,
        scheduler_state_dict: Dict,
        strict: bool = True,
    ):
        if self.distributed:
            self.model.module.load_state_dict(model_state_dict, strict=strict)
        else:
            self.model.load_state_dict(model_state_dict, strict=strict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

    def compute_fisher_matrix(self, loader: Iterable, loss_thr: float):
        fisher_information = {}
        for name, _param in self.model.named_parameters():
            fisher_information[name] = torch.zeros_like(_param)

        self.model.train()
        cnt_updated = 0
        for _, batch in enumerate(loader):
            self.model.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            total_loss = torch.tensor([0.0], device=self.device)
            for loss_def, w in self.loss_functions:
                total_loss += loss_def.get_loss(output, self.model) * w
            if loss_thr < 0 or total_loss < loss_thr:
                total_loss.backward()
                for name, _param in self.model.named_parameters():
                    if _param.grad is not None:
                        fisher_information[name] += _param.grad.detach().clone() ** 2
                cnt_updated += 1

        for name in fisher_information:
            fisher_information[name] /= cnt_updated

        optimal_params =\
            {k: v.data.detach().clone() for k, v in self.model.named_parameters()}
        return fisher_information, optimal_params, cnt_updated


class RehearsalTrainer(Trainer):

    def run_one_epoch_rehearsal(
        self,
        loader,
        memloader,
        is_train: bool = False,
        error_recorder: Optional[ErrorRecorder] = None,
        mem_recorder: Optional[ErrorRecorder] = None,
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Defile memory iterator that uses in
        mem_iter = iter(memloader)

        for _, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)

            output = self.model(batch)

            if is_train:
                total_loss = torch.tensor([0.0], device=self.device)
                for loss_def, w in self.loss_functions:
                    total_loss += loss_def.get_loss(output, self.model) * w

                total_loss.backward()
                self.optimizer.step()

            if error_recorder is not None:
                error_recorder.update(output)
                error_recorder.update_not_only_data(
                    self.model, self.loss_functions, output
                )

            # Fetch mini-batch from memory iterator
            try:
                batch_mem = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memloader)  # Recreate the iterator if exhausted
                batch_mem = next(mem_iter)

            batch_mem = batch_mem.to(self.device, non_blocking=True)
            memout = self.model(batch_mem)

            if is_train:
                mem_loss = torch.tensor([0.0], device=self.device)
                for loss_def, w in self.loss_functions:
                    mem_loss += loss_def.get_loss(memout, self.model) * w

                mem_loss.backward()
                self.optimizer.step()

            if mem_recorder is not None:
                mem_recorder.update(memout)
                mem_recorder.update_not_only_data(
                    self.model, self.loss_functions, memout
                )

        if self.distributed:
            if error_recorder is not None:
                self.recorder_all_reduce(error_recorder)
            if mem_recorder is not None:
                self.recorder_all_reduce(mem_recorder)
