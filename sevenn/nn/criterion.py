import torch


class VectorHuberLoss(torch.nn.HuberLoss):

    def __init__(self, vdim, **kwargs):
        super().__init__(**kwargs)
        assert self.reduction in ['none', 'mean']
        self.vdim = vdim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _inp = input.view(-1, self.vdim)
        _tg = target.view(-1, self.vdim)
        mse = (_inp - _tg).square().sum(1)
        d = (mse + 1e-6).sqrt()
        mask = d < self.delta
        out = torch.where(
            mask,
            mse * 0.5 * self.delta,
            self.delta * (d - 0.5 * self.delta),
        )
        if self.reduction == 'mean':
            return out.mean()
        else:
            return out
