from typing import Callable, List, Optional

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class IrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        is_embed: bool = False,
        **e3nn_linear_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        self._last_dim = irreps_in.dim
        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)
        if is_embed:
            # eliminate path weight of e3nn, as it is used for embedding
            # e3nn initialize weight using torch.randn, therefore our var is ~= 1
            assert irreps_in.lmax == 0 and irreps_out.lmax == 0
            self.linear.weight =\
                torch.nn.Parameter(
                    self.linear.weight / self.linear.instructions[0].path_weight
                )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_input].reshape(-1, self._last_dim)
        data[self.key_output] = self.linear(x)
        return data


@compile_mode('script')
class AtomReduce(nn.Module):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self,
        data_key_in: str,
        data_key_out: str,
        reduce: str = 'sum',
        constant: float = 1.0,
    ):
        super().__init__()

        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = constant
        self.reduce = reduce

        # controlled by the upper most wrapper 'AtomGraphSequential'
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            src = data[self.key_input].squeeze(1)
            size = int(data[KEY.BATCH].max()) + 1
            output = torch.zeros(
                (size), dtype=src.dtype, device=src.device,
            )
            output.scatter_reduce_(0, data[KEY.BATCH], src, reduce='sum')
            data[self.key_output] = output * self.constant
        else:
            data[self.key_output] = (
                torch.sum(data[self.key_input]) * self.constant
            )

        return data


@compile_mode('script')
class FCN_e3nn(nn.Module):
    """
    wrapper class of e3nn FullyConnectedNet
    """

    def __init__(
        self,
        irreps_in: Irreps,  # confirm it is scalar & input size
        dim_out: int,
        hidden_neurons: List[int],
        activation: Callable,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        **e3nn_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.irreps_in = irreps_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        for _, irrep in irreps_in:
            assert irrep.is_scalar()
        inp_dim = irreps_in.dim

        self.fcn = FullyConnectedNet(
            [inp_dim] + hidden_neurons + [dim_out],
            activation,
            **e3nn_params,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.fcn(data[self.key_input])
        return data


@compile_mode('script')
class IrrepsStrideLinear(nn.Module):
    """
    Stride version of linear
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: Optional[str] = None,
        unsqueeze_scalar: bool = True,
        **e3nn_linear_params,
    ):
        super().__init__()
        assert all([irreps_in[0].mul == mul for mul, _ in irreps_in])
        multiplicity = irreps_in[0].mul
        # TODO: changing channel length is possible for stride
        assert all([mul == multiplicity for mul, _ in irreps_out])
        for irr in [irreps_in, irreps_out]:
            assert irr == irr.sort().irreps.simplify()

        self.mul_in: int = multiplicity
        self.mul_out: int = irreps_out[0].mul
        self.ir_dim_in = (irreps_in.lmax + 1)**2
        self.ir_dim_out = (irreps_out.lmax + 1)**2
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        if unsqueeze_scalar:
            self.unsqueeze_scalar = irreps_in.lmax == 0
        else:
            self.unsqueeze_scalar = False

        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        e3nn_linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)
        assert len(e3nn_linear.instructions) != 0, 'No instructions given'

        weights = nn.ParameterList()
        for idx, inst in enumerate(e3nn_linear.instructions):
            w = e3nn_linear.weight_view_for_instruction(idx).clone().contiguous()
            # It becomes slightly difference from e3nn
            weights.append(w * inst.path_weight)
        self.weights = weights

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_input]
        if self.unsqueeze_scalar:
            # last two dimension should be (mul, ir_dim)
            # If x is pure scalar from e3nn, it can be seen as strided
            x = x.unsqueeze(-1)

        in_size = self.mul_in * self.ir_dim_in
        out_size = self.mul_out * self.ir_dim_out
        output_shape = x.shape[:-2] + (self.mul_out, self.ir_dim_out)

        w = torch.zeros(
            size=(in_size, out_size),
            device=x.device,
            dtype=x.dtype
        )
        prev = 0
        for l, weight in enumerate(self.weights):
            for i in range(2 * l + 1):
                offset = prev + i
                w[offset::self.ir_dim_in, offset::self.ir_dim_out] = weight
            prev += 2 * l + 1
        data[self.key_output] = (x.reshape(-1, in_size) @ w).reshape(output_shape)

        return data
