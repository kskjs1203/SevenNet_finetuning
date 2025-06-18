from typing import Any, Callable, Dict, Optional

import torch

import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None,
        vdim: Optional[int] = None,
        use_weight: bool = False,  # use data dependent weight
        weight_key: Optional[str] = None,
        delete_unlabled: bool = True,
    ):
        if criterion is not None and hasattr(criterion, 'reduction'):
            if use_weight:
                assert criterion.reduction == 'none'
                assert weight_key is not None
            else:
                assert criterion.reduction != 'none'

        assert isinstance(vdim, int)
        self.name = name
        self.unit = unit
        self.vdim = vdim
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.weight_key = weight_key
        self.delete_unlabeled = delete_unlabled

    def assign_criteria(self, criterion: Callable):
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        return torch.reshape(batch_data[self.pred_key], (-1,)), torch.reshape(
            batch_data[self.ref_key], (-1,)
        )

    def _get_data_weight(self, batch_data):
        return torch.repeat_interleave(
            batch_data[KEY.DATA_WEIGHT][self.weight_key], self.vdim
        )

    def get_loss(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        pred, ref = self._preprocess(batch_data, model)
        weights = None
        if self.use_weight:
            weights = self._get_data_weight(batch_data)

        if self.delete_unlabeled:
            unlabeled_idx = torch.isnan(ref)
            pred = pred[~unlabeled_idx]
            ref = ref[~unlabeled_idx]
            if len(pred) == 0:
                return torch.zeros(1, device=pred.device)
            if self.use_weight:
                assert isinstance(weights, list)
                weights = weights[~unlabeled_idx]

        if self.use_weight:
            return torch.mean(self.criterion(pred, ref) * weights)
        else:
            return self.criterion(pred, ref)


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_TOTAL_ENERGY,
        weight_key: Optional[str] = KEY.PER_ATOM_ENERGY,
        **kwargs
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            weight_key=weight_key,
            vdim=1,
            **kwargs
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            batch_data[self.pred_key] / num_atoms,
            batch_data[self.ref_key] / num_atoms,
        )


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
        weight_key: Optional[str] = KEY.FORCE,
        **kwargs,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            vdim=3,
            weight_key=weight_key,
            **kwargs,
        )

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key], (-1,)),
            torch.reshape(batch_data[self.ref_key], (-1,)),
        )

    def _get_data_weight(self, batch_data):
        weight = batch_data[KEY.DATA_WEIGHT][self.weight_key]
        weight_tensor = weight[batch_data[KEY.BATCH]]
        return torch.repeat_interleave(weight_tensor, self.vdim)


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
        weight_key: Optional[str] = KEY.STRESS,
        **kwargs,
    ):
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            vdim=6,
            weight_key=weight_key,
            **kwargs,
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        return (
            torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,)),
            torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,)),
        )


class EWCLoss(LossDefinition):

    def __init__(
        self,
        fisher_dict: Dict,
        opt_params_dict: Dict,
        name: str = 'EWC',
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            criterion=None,
            name=name,
            ref_key=None,
            pred_key=None,
            weight_key=None,
            use_weight=False,
            vdim=0,
            **kwargs
        )
        self.fisher_dict = fisher_dict
        self.opt_params_dict = opt_params_dict
        self.device = device
        if device is not None:
            self.to(device)

    def to(self, device):
        for k in self.fisher_dict:
            self.fisher_dict[k] = self.fisher_dict[k].to(device)
        for k in self.opt_params_dict:
            self.opt_params_dict[k] = self.opt_params_dict[k].to(device)
        self.device = device

    def get_loss(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ):
        _ = batch_data
        if model is None:
            raise ValueError('EWC requires model to compute loss')
        ewc_loss = torch.tensor([0.0], device=self.device)
        for name, _param in model.named_parameters():
            if name not in self.fisher_dict or name not in self.opt_params_dict:
                continue
            fisher = self.fisher_dict[name]
            opt_param = self.opt_params_dict[name]
            ewc_loss += torch.sum(fisher * (_param - opt_param) ** 2)
        return ewc_loss


def get_loss_functions_from_config(config):
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    try:
        loss_param = config[KEY.LOSS_PARAM]
    except KeyError:
        loss_param = {}

    if loss == 'custom':
        return _loss_functions_from_callback(config, **loss_param)

    reduction = 'mean'
    use_weight = False
    if KEY.LOAD_DATASET_WITH_WEIGHTS in config:
        reduction = 'none'
        use_weight = True
    criterion = loss(reduction=reduction, **loss_param)
    common = {
        'criterion': criterion,
        'use_weight': use_weight,
    }

    loss_functions.append((PerAtomEnergyLoss(**common), 1.0))
    loss_functions.append((ForceLoss(**common), config[KEY.FORCE_WEIGHT]))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(**common), config[KEY.STRESS_WEIGHT]))

    fisher_information_path = config[KEY.CONTINUE][KEY.FISHER]
    optimal_params_path = config[KEY.CONTINUE][KEY.OPT_PARAMS]
    if fisher_information_path is not False and optimal_params_path is not False:
        fisher_dict = torch.load(fisher_information_path, weights_only=True)
        opt_params_dict = torch.load(optimal_params_path, weights_only=True)
        ewc_lambda = float(config[KEY.CONTINUE][KEY.EWC_LAMBDA])
        device = config[KEY.DEVICE]
        loss_functions.append(
            (EWCLoss(fisher_dict, opt_params_dict, device=device), ewc_lambda / 2.0)
        )

    return loss_functions


def _loss_functions_from_callback(config, path, module, function):
    import importlib
    import os
    import sys
    if not os.path.isdir(path):
        raise ValueError(f'No such dir: {path}')
    sys.path.insert(1, path)
    mm = importlib.import_module(module)
    _loss_f_callback = getattr(mm, function)
    return _loss_f_callback(config)
