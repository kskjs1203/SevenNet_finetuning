from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

import sevenn._keys as KEY

from .atom_graph_data import AtomGraphData
from .train.optim import loss_dict

ERROR_TYPES = {
    'TotalEnergy': {
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV',
        'vdim': 1,
    },
    'Energy': {  # by default per-atom for energy
        'name': 'Energy',
        'ref_key': KEY.ENERGY,
        'pred_key': KEY.PRED_TOTAL_ENERGY,
        'unit': 'eV/atom',
        'per_atom': True,
        'vdim': 1,
    },
    'Force': {
        'name': 'Force',
        'ref_key': KEY.FORCE,
        'pred_key': KEY.PRED_FORCE,
        'unit': 'eV/Å',
        'vdim': 3,
    },
    'Stress': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'kbar',
        'coeff': 1602.1766208,
        'vdim': 6,
    },
    'Stress_GPa': {
        'name': 'Stress',
        'ref_key': KEY.STRESS,
        'pred_key': KEY.PRED_STRESS,
        'unit': 'GPa',
        'coeff': 160.21766208,
        'vdim': 6,
    },
    'TotalLoss': {
        'name': 'TotalLoss',
        'unit': None,
    },
    'EWCLoss': {
        'name': 'EWC',
        'unit': None,
    }
}


class AverageNumber:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, values: torch.Tensor):
        self._sum += values.sum().item()
        self._count += values.numel()

    def _ddp_reduce(self, device):
        _sum = torch.tensor(self._sum, device=device)
        _count = torch.tensor(self._count, device=device)
        dist.all_reduce(_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(_count, op=dist.ReduceOp.SUM)
        self._sum = _sum.item()
        self._count = _count.item()

    def get(self):
        if self._count == 0:
            return torch.nan
        return self._sum / self._count


class ErrorMetric:
    """
    Base class for error metrics We always average error by # of structures,
    and designed to collect errors in the middle of iteration (by AverageNumber)
    """

    def __init__(
        self,
        name: str,
        ref_key: str,
        pred_key: str,
        coeff: float = 1.0,
        unit: Optional[str] = None,
        per_atom: bool = False,
        delete_unlabled: bool = True,
        **kwargs,
    ):
        self.name = name
        self.unit = unit
        self.coeff = coeff
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.per_atom = per_atom
        self.delete_unlabled = delete_unlabled
        self.value = AverageNumber()

    def update(self, output: AtomGraphData):
        raise NotImplementedError

    def _retrieve(self, output: AtomGraphData):
        y_ref = output[self.ref_key] * self.coeff
        y_pred = output[self.pred_key] * self.coeff
        if self.per_atom:
            natoms = output[KEY.NUM_ATOMS]
            y_ref = y_ref / natoms
            y_pred = y_pred / natoms
        if self.delete_unlabled:
            unlabelled_idx = torch.isnan(y_ref)
            y_ref = y_ref[~unlabelled_idx]
            y_pred = y_pred[~unlabelled_idx]
        return y_ref, y_pred

    def ddp_reduce(self, device):
        self.value._ddp_reduce(device)

    def reset(self):
        self.value = AverageNumber()

    def get(self):
        return self.value.get()

    def key_str(self):
        if self.unit is None:
            return self.name
        else:
            return f'{self.name} ({self.unit})'

    def __str__(self):
        return f'{self.key_str()}: {self.value.get():.6f}'


class RMSError(ErrorMetric):
    """
    Vector squared error
    """

    def __init__(self, vdim: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.vdim = vdim
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(self, y_ref, y_pred, vdim: int):
        return self._se(y_ref, y_pred).view(-1, vdim).sum(dim=1)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        se = self._square_error(y_ref, y_pred, self.vdim)
        self.value.update(se)

    def get(self):
        return self.value.get() ** 0.5


class ComponentRMSError(ErrorMetric):
    """
    Ignore vector dim and just average over components
    Results in smaller error looking
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._se = torch.nn.MSELoss(reduction='none')

    def _square_error(self, y_ref, y_pred):
        return self._se(y_ref, y_pred)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.view(-1)
        y_pred = y_pred.view(-1)
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)

    def get(self):
        return self.value.get() ** 0.5


class MAError(ErrorMetric):
    """
    Average over all component
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _square_error(self, y_ref, y_pred):
        return torch.abs(y_ref - y_pred)

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.reshape((-1,))
        y_pred = y_pred.reshape((-1,))
        se = self._square_error(y_ref, y_pred)
        self.value.update(se)


class VectorMAError(ErrorMetric):
    """
    Average of distance between two vectors
    Aims to work with force error.
    This is rotationally invariant
    """

    def __init__(self, vdim: int = 3, **kwargs):
        self.vdim = vdim
        super().__init__(**kwargs)

    def _square_error(self, y_ref, y_pred):
        _v = torch.abs(y_ref - y_pred).square().view(-1, self.vdim)
        return _v.sum(dim=1).sqrt()

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)

        se = self._square_error(y_ref, y_pred)
        self.value.update(se)

    def get(self):
        return self.value.get()


class CustomError(ErrorMetric):
    """
    Custom error metric
    Args:
        func: a function that takes y_ref and y_pred
              and returns a list of errors
    """

    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def update(self, output: AtomGraphData):
        y_ref, y_pred = self._retrieve(output)
        se = self.func(y_ref, y_pred) if len(y_ref) > 0 else torch.tensor([])
        self.value.update(se)


class CombinedError(ErrorMetric):
    """
    Combine multiple error metrics with weights
    corresponds to a weighted sum of errors (normally used in loss)
    """

    def __init__(self, metrics: List[Tuple[ErrorMetric, float]], **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        assert kwargs['unit'] is None

    def update(self, output: AtomGraphData):
        for metric, _ in self.metrics:
            metric.update(output)

    def reset(self):
        for metric, _ in self.metrics:
            metric.reset()

    def ddp_reduce(self, device):  # override
        for metric, _ in self.metrics:
            metric.value._ddp_reduce(device)

    def get(self):
        val = 0.0
        for metric, weight in self.metrics:
            val += metric.get() * weight
        return val


# better to create another base class... but only for this error
class EWCError(ErrorMetric):

    def __init__(self, ewc_loss_name: str = 'EWC', **kwargs):
        super().__init__(ref_key='', pred_key='', **kwargs)
        self._update_not_only_weight_flag = True
        self.ewc_loss_name = ewc_loss_name
        self.value = None  # no average number

    def update(self, output: AtomGraphData):
        _ = output
        raise NotImplementedError()

    def update_not_only_data(self, model, loss_functions, output: AtomGraphData):
        if self.value is not None:
            return  # EWC loss no need average over steps
        for loss_def, w in loss_functions:
            if loss_def.name != self.ewc_loss_name:
                continue
            self.value = loss_def.get_loss(output, model) * w

    def get(self):
        if self.value is None:
            return torch.nan
        return self.value.item()

    def reset(self):
        self.value = None

    def ddp_reduce(self, device):
        _ = device
        return


class ErrorRecorder:
    """
    record errors of a model
    """

    METRIC_DICT = {
        'RMSE': RMSError,
        'ComponentRMSE': ComponentRMSError,
        'VectorMAE': VectorMAError,
        'MAE': MAError,
        'Loss': CustomError,
    }

    def __init__(self, metrics: List[ErrorMetric]):
        self.history = []
        self.metrics = metrics

        self.special_metrics = []  # dups of metrics of special
        for metric in metrics:
            if hasattr(metric, '_update_not_only_weight_flag') \
                    and metric._update_not_only_weight_flag:
                self.special_metrics.append(metric)

    def _update(self, output: AtomGraphData):
        for metric in self.metrics:
            if hasattr(metric, '_update_not_only_weight_flag') \
                    and metric._update_not_only_weight_flag:
                continue
            metric.update(output)

    def update(self, output: AtomGraphData, no_grad: bool = True):
        if no_grad:
            with torch.no_grad():
                self._update(output)
        else:
            self._update(output)

    def update_not_only_data(
        self, model, loss_functions, output: AtomGraphData
    ):
        for metric in self.special_metrics:
            assert hasattr(metric, 'update_not_only_data')
            metric.update_not_only_data(model, loss_functions, output)

    def get_metric_dict(self):
        return {metric.key_str(): metric.get() for metric in self.metrics}

    def epoch_forward(self):
        self.history.append(self.get_metric_dict())
        for metric in self.metrics:
            metric.reset()
        return self.history[-1]

    def get_history(self):
        return self.history

    @staticmethod
    def init_total_loss_metric(config, criteria):
        is_stress = config[KEY.IS_TRAIN_STRESS]
        metrics = []
        energy_metric = CustomError(criteria, **ERROR_TYPES['Energy'])
        metrics.append((energy_metric, 1))
        force_metric = CustomError(criteria, **ERROR_TYPES['Force'])
        metrics.append((force_metric, config[KEY.FORCE_WEIGHT]))
        if is_stress:
            stress_metric = CustomError(criteria, **ERROR_TYPES['Stress'])
            metrics.append((stress_metric, config[KEY.STRESS_WEIGHT]))
        total_loss_metric = CombinedError(
            metrics, name='TotalLoss', unit=None, ref_key=None, pred_key=None
        )
        return total_loss_metric

    @staticmethod
    def init_ewc_loss_metric(config):
        # TODO: better check
        assert float(config[KEY.CONTINUE][KEY.EWC_LAMBDA]) != 0
        return EWCError(**ERROR_TYPES['EWCLoss'])

    @staticmethod
    def from_config(config: dict):
        specials = ['TotalLoss', 'EWCLoss']
        loss_cls = loss_dict[config[KEY.LOSS].lower()]
        try:
            loss_param = config[KEY.LOSS_PARAM]
        except KeyError:
            loss_param = {}
        criteria = loss_cls(**loss_param)

        err_config = config[KEY.ERROR_RECORD]
        err_config_n = []
        if not config[KEY.IS_TRAIN_STRESS]:
            for err_type, metric_name in err_config:
                if 'Stress' in err_type:
                    continue
                err_config_n.append((err_type, metric_name))
            err_config = err_config_n

        err_metrics = []
        for err_type, metric_name in err_config:
            metric_kwargs = ERROR_TYPES[err_type].copy()
            if err_type in specials:
                metric = None
                if err_type == 'TotalLoss':
                    metric = ErrorRecorder.init_total_loss_metric(config, criteria)
                elif err_type == 'EWCLoss':
                    metric = ErrorRecorder.init_ewc_loss_metric(config)
                err_metrics.append(metric)
            else:
                metric_cls = ErrorRecorder.METRIC_DICT[metric_name]
                metric_kwargs['name'] += f'_{metric_name}'
                if metric_name == 'Loss':
                    metric_kwargs['func'] = criteria
                    metric_kwargs['unit'] = None
                err_metrics.append(metric_cls(**metric_kwargs))

        return ErrorRecorder(err_metrics)
