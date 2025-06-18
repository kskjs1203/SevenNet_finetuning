from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
import sevenn.nn.gaunt_util as gutil
from sevenn._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .equivariant_product_basis import reshape_irreps
from .util import _broadcast


def message_gather(
    node_features: torch.Tensor,
    edge_dst: torch.Tensor,
    message: torch.Tensor
):
    index = _broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape,
        dtype=node_features.dtype,
        device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        weight_shift: float = 0.0,
        weight_scale: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
    ):
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx

        self.w_shift = weight_shift
        self.w_scale = weight_scale

        self.is_parallel = is_parallel

        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self.convolution = TensorProduct(
            irreps_x,
            irreps_filter,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn = FullyConnectedNet(
            weight_layer_input_to_hidden + [self.convolution.weight_numel],
            weight_layer_act,
        )

        self._comm_size = self.convolution.irreps_in1.dim  # used in parallel

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        weight_in = self.w_scale * (data[self.key_weight_input] - self.w_shift)
        weight = self.weight_nn(weight_in)
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        message = self.convolution(x[edge_src], data[self.key_filter], weight)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)
        if self.is_parallel:
            # NLOCAL is # of atoms in system at 'CPU'
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data


@compile_mode('script')
class GauntConvolution(nn.Module):

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        weight_shift: float = 0.0,
        weight_scale: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
        fit_gaunt_to_w3j_mode: str = 'norm',
        fft_norm: str = 'forward',
        is_stride: bool = False,
        _compute_edge_fft: bool = True,
        _init_edge_fourier: bool = True,
    ):
        super().__init__()
        # multiplicity should be same for all irrep in irreps_x and out
        assert all([irreps_x[0].mul == mul for mul, _ in irreps_x])
        multiplicity = irreps_x[0].mul
        assert all([multiplicity == mul for mul, _ in irreps_out])
        assert all([1 == mul for mul, _ in irreps_filter])
        for irr in [irreps_x, irreps_filter, irreps_out]:
            assert irr == irr.sort().irreps.simplify()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx

        self.irreps_x = irreps_x
        self.irreps_filter = irreps_filter
        self.irreps_out = irreps_out

        self.fft_norm = fft_norm

        self.w_shift = weight_shift
        self.w_scale = weight_scale

        self.is_stride = is_stride
        self.is_parallel = is_parallel

        self._layout_convert_x = reshape_irreps(irreps_x)
        self._layout_convert_out = reshape_irreps(irreps_out)
        self._compute_edge_fft = _compute_edge_fft
        self._init_edge_fourier = _init_edge_fourier

        a_wo = gutil.weight_align_matrix(irreps_out.lmax)
        # apply e3nn path_weights
        path_w = torch.FloatTensor([ir.dim**0.5 for _, ir in self.irreps_out])
        if fit_gaunt_to_w3j_mode != 'none':
            path_w *= gutil.fit_gaunt_to_w3j(  # Gaunt vs CGTP ratio
                irreps_x.lmax,
                irreps_filter.lmax,
                fit_gaunt_to_w3j_mode
            )[:irreps_out.lmax + 1]
            a_wo = (a_wo.T * path_w).T
        self.register_buffer('a_w', a_wo)
        weight_shape = torch.Size((multiplicity, len(irreps_out)))

        self.weight_shape = weight_shape

        self.weight_nn = FullyConnectedNet(
            weight_layer_input_to_hidden + [weight_shape.numel()],
            weight_layer_act,
        )

        self._x_to_fourier = gutil.ToFourierBasis(irreps_x)
        self._filter_to_fourier = gutil.ToFourierBasis(irreps_filter)
        self._L_true = irreps_x.lmax + irreps_filter.lmax
        self._out_to_sph = gutil.ToSphericalBasis(self._L_true, irreps_out.lmax)

        self._comm_size = self.irreps_x.dim  # used in parallel

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        if not self.is_stride:
            x = self._layout_convert_x(x)

        x_fourier = self._x_to_fourier(x)
        weight_in = self.w_scale * (data[self.key_weight_input] - self.w_shift)
        weight = self.weight_nn(weight_in)

        L = self._L_true
        size = (2 * L + 1, 2 * L + 1)

        if self._init_edge_fourier:
            # multiplicity dim(=1) for filter
            filter = data[self.key_filter].unsqueeze(1)
            data['edge_fourier'] =\
                self._filter_to_fourier(filter)

        if self._compute_edge_fft:
            filter_f =\
                torch.fft.fft2(data['edge_fourier'], s=size, norm=self.fft_norm)
            data[f'edge_fft_{L}'] = filter_f
        else:
            filter_f = data[f'edge_fft_{L}']

        x_f = torch.fft.fft2(x_fourier, s=size, norm=self.fft_norm)
        _conv = x_f[edge_src] * filter_f
        _ffted = torch.fft.ifft2(_conv, norm=self.fft_norm)
        _message = self._out_to_sph(_ffted)
        message = _message * (weight.reshape((-1,) + self.weight_shape) @ self.a_w)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)

        if not self.is_stride:
            x = self._layout_convert_out.reverse(x)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data


@compile_mode('script')
class HermitianGauntConvolution(nn.Module):

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        weight_shift: float = 0.0,
        weight_scale: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
        fit_gaunt_to_w3j_mode: str = 'norm',
        fft_norm: str = 'backward',
        is_stride: bool = False,
        _compute_edge_fft: bool = True,
        _init_edge_fourier: bool = True,
    ):
        super().__init__()
        # multiplicity should be same for all irrep in irreps_x and out
        assert all([irreps_x[0].mul == mul for mul, _ in irreps_x])
        multiplicity = irreps_x[0].mul
        assert all([multiplicity == mul for mul, _ in irreps_out])
        assert all([1 == mul for mul, _ in irreps_filter])
        for irr in [irreps_x, irreps_filter, irreps_out]:
            assert irr == irr.sort().irreps.simplify()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx

        self.irreps_x = irreps_x
        self.irreps_filter = irreps_filter
        self.irreps_out = irreps_out

        self.fft_norm = fft_norm

        self.w_shift = weight_shift
        self.w_scale = weight_scale

        self.is_stride = is_stride
        self.is_parallel = is_parallel

        self._layout_convert_x = reshape_irreps(irreps_x)
        self._layout_convert_out = reshape_irreps(irreps_out)
        self._compute_edge_fft = _compute_edge_fft
        self._init_edge_fourier = _init_edge_fourier

        a_wo = gutil.weight_align_matrix(irreps_out.lmax)
        # apply e3nn path_weights
        path_w = torch.FloatTensor([ir.dim**0.5 for _, ir in self.irreps_out])
        if fit_gaunt_to_w3j_mode != 'none':
            path_w *= gutil.fit_gaunt_to_w3j(  # Gaunt vs CGTP ratio
                irreps_x.lmax,
                irreps_filter.lmax,
                fit_gaunt_to_w3j_mode
            )[:irreps_out.lmax + 1]
            a_wo = (a_wo.T * path_w).T
        self.register_buffer('a_w', a_wo)
        weight_shape = torch.Size((multiplicity, len(irreps_out)))

        self.weight_shape = weight_shape

        self.weight_nn = FullyConnectedNet(
            weight_layer_input_to_hidden + [weight_shape.numel()],
            weight_layer_act,
        )

        self._x_to_fourier = gutil.ToFourierBasis(irreps_x)
        self._filter_to_fourier = gutil.ToFourierBasis(irreps_filter)
        self._L_true = irreps_x.lmax + irreps_filter.lmax
        self._out_to_sph = gutil.ToSphericalBasis(
            self._L_true, irreps_out.lmax, hermitian_input=True
        )

        self._comm_size = self.irreps_x.dim  # used in parallel

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        uv_dim1 = 2 * self.irreps_x.lmax + 1
        uv_dim2 = 2 * self.irreps_filter.lmax + 1
        conv_dim = uv_dim1 + uv_dim2 - 1

        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        L = self._L_true
        if self._init_edge_fourier:
            filter = data[self.key_filter].unsqueeze(1)
            data['edge_fourier'] = self._filter_to_fourier(filter)
        if self._compute_edge_fft:
            y_fourier = data['edge_fourier']
            pad_y = (conv_dim - uv_dim2) // 2
            y_fourier_p = F.pad(y_fourier, (pad_y, pad_y, pad_y, pad_y))
            y_fourier_onesided =\
                torch.fft.ifftshift(y_fourier_p)[..., :conv_dim // 2 + 1]
            yf = torch.fft.irfft2(
                y_fourier_onesided, s=(conv_dim, conv_dim), norm=self.fft_norm
            )
            data[f'edge_fft_{L}'] = yf
        yf = data[f'edge_fft_{L}']

        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])
        if not self.is_stride:
            x = self._layout_convert_x(x)
        x_fourier = self._x_to_fourier(x)
        pad_x = (conv_dim - uv_dim1) // 2
        x_fourier_p = F.pad(x_fourier, (pad_x, pad_x, pad_x, pad_x))
        x_fourier_onesided =\
            torch.fft.ifftshift(x_fourier_p)[..., :conv_dim // 2 + 1]

        xf = torch.fft.irfft2(
            x_fourier_onesided, s=(conv_dim, conv_dim), norm=self.fft_norm
        )

        conv = xf[edge_src] * yf
        out_fourier = gutil.rfft2(conv)
        _message = self._out_to_sph(out_fourier)
        weight_in = self.w_scale * (data[self.key_weight_input] - self.w_shift)
        weight = self.weight_nn(weight_in)
        message = _message * (weight.reshape((-1,) + self.weight_shape) @ self.a_w)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)

        if not self.is_stride:
            x = self._layout_convert_out.reverse(x)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data
