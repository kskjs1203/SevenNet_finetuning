import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
import sevenn.nn.gaunt_util as gutil
from sevenn._const import AtomGraphDataType
from sevenn.nn.equivariant_product_basis import reshape_irreps


@compile_mode('script')
class GauntProductBasisWithWeight(nn.Module):
    """
    Gaunt version of equivariant product basis
    tensor product of itself v times, no weights
    """

    def __init__(
        self,
        irreps_x: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_attr: str = KEY.NODE_ATTR,
        fit_gaunt_to_w3j_mode: str = 'norm',
        fft_norm: str = 'ortho',
        is_stride: bool = False,
    ):
        super().__init__()
        assert all([irreps_x[0].mul == mul for mul, _ in irreps_x])
        multiplicity = irreps_x[0].mul
        assert all([multiplicity == mul for mul, _ in irreps_out])
        for irr in [irreps_x, irreps_out]:
            assert irr == irr.sort().irreps.simplify()

        self._L_true = correlation * irreps_x.lmax
        assert irreps_out.lmax <= self._L_true

        self._L_x = irreps_x.lmax
        self.correlation = correlation

        self.irreps_x = irreps_x
        self.irreps_out = irreps_out
        self.key_x = data_key_x
        self.key_attr = data_key_attr

        self.fft_norm = fft_norm
        self.is_stride = is_stride

        weight_shape = (multiplicity, irreps_x.lmax + 1)
        self.weights = torch.nn.ParameterList([torch.randn(weight_shape)])
        self.register_buffer('a_w', gutil.weight_align_matrix(irreps_x.lmax))
        for _ in range(2, self.correlation + 1):
            self.weights.append(torch.randn(weight_shape))

        # e3nn normalization
        path_w = torch.FloatTensor([ir.dim**0.5 for _, ir in self.irreps_out])
        if fit_gaunt_to_w3j_mode != 'none':
            ratio = torch.ones(irreps_out.lmax + 1)
            base_l = irreps_x.lmax
            for _ in range(self.correlation - 1):
                _r = gutil.fit_gaunt_to_w3j(
                    base_l, irreps_x.lmax, fit_gaunt_to_w3j_mode
                )
                len_ = min(len(_r), len(ratio))
                ratio[:len_] *= _r[:len_]
                base_l += irreps_x.lmax
            path_w *= ratio
        indi = []
        for l in range(irreps_out.lmax + 1):
            for _ in range(2 * l + 1):
                indi.append(l)
        path_w = path_w[indi].contiguous()
        self.register_buffer('path_w', path_w)

        self._layout_convert_x = reshape_irreps(irreps_x)
        self._layout_convert_out = reshape_irreps(irreps_out)

        self._x_to_fourier = gutil.ToFourierBasis(irreps_x)
        self._out_to_sph = gutil.ToSphericalBasis(self._L_true, irreps_out.lmax)
        self.complex_dtype = self._x_to_fourier.complex_dtype

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if not self.is_stride:
            x = self._layout_convert_x(x)

        # last dimension is about irreps, except for it
        outer_shape = x.shape[:-1]
        L_x = self._L_x
        L_out = self._L_true
        size = 2 * L_out + 1

        out = torch.zeros(
            (outer_shape + (size, size)),
            dtype=self.complex_dtype,
            device=x.device,
        )

        x_0 = self._x_to_fourier(x * (self.weights[0] @ self.a_w))
        x_f = torch.fft.fft2(x_0, s=(size, size), norm=self.fft_norm)

        base = x_f
        c = L_out - L_x
        r = 2 * L_x + 1
        out[..., c:c + r, c:c + r] =\
            out[..., c:c + r, c:c + r] + x_0[..., :r, :r]

        for i, v in enumerate(range(2, self.correlation + 1)):
            c = L_out - L_x * v
            r = 2 * L_x * v + 1

            x_v = self._x_to_fourier(x * (self.weights[i + 1] @ self.a_w))
            x_f = torch.fft.fft2(x_v, s=(size, size), norm=self.fft_norm)

            base = base * x_f
            out[..., c:c + r, c:c + r] =\
                out[..., c:c + r, c:c + r] +\
                torch.fft.ifft2(base, norm=self.fft_norm)[..., :r, :r]

        x = self._out_to_sph(out)  # [..., (L_out+1)**2]
        x = x * self.path_w

        if not self.is_stride:
            x = self._layout_convert_out.reverse(x)

        data[self.key_x] = x
        return data


@compile_mode('script')
class GauntProductBasis(nn.Module):
    """
    Gaunt version of equivariant product basis
    tensor product of itself v times, no weights
    """

    def __init__(
        self,
        irreps_x: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_attr: str = KEY.NODE_ATTR,
        fit_gaunt_to_w3j_mode: str = 'norm',
        fft_norm: str = 'ortho',
        is_stride: bool = False,
    ):
        super().__init__()
        assert all([irreps_x[0].mul == mul for mul, _ in irreps_x])
        multiplicity = irreps_x[0].mul
        assert all([multiplicity == mul for mul, _ in irreps_out])
        for irr in [irreps_x, irreps_out]:
            assert irr == irr.sort().irreps.simplify()

        self._L_true = correlation * irreps_x.lmax
        assert irreps_out.lmax <= self._L_true

        self._L_x = irreps_x.lmax
        self.correlation = correlation

        self.irreps_x = irreps_x
        self.irreps_out = irreps_out
        self.key_x = data_key_x
        self.key_attr = data_key_attr

        self.fft_norm = fft_norm
        self.is_stride = is_stride

        # e3nn normalization
        path_w = torch.FloatTensor([ir.dim**0.5 for _, ir in self.irreps_out])
        if fit_gaunt_to_w3j_mode != 'none':
            ratio = torch.ones(irreps_out.lmax + 1)
            base_l = irreps_x.lmax
            for _ in range(self.correlation - 1):
                _r = gutil.fit_gaunt_to_w3j(
                    base_l, irreps_x.lmax, fit_gaunt_to_w3j_mode
                )
                len_ = min(len(_r), len(ratio))
                ratio[:len_] *= _r[:len_]
                base_l += irreps_x.lmax
            path_w *= ratio
        indi = []
        for l in range(irreps_out.lmax + 1):
            for _ in range(2 * l + 1):
                indi.append(l)
        path_w = path_w[indi].contiguous()
        self.register_buffer('path_w', path_w)

        self._layout_convert_x = reshape_irreps(irreps_x)
        self._layout_convert_out = reshape_irreps(irreps_out)

        self._x_to_fourier = gutil.ToFourierBasis(irreps_x)
        self._out_to_sph = gutil.ToSphericalBasis(self._L_true, irreps_out.lmax)
        self.complex_dtype = self._x_to_fourier.complex_dtype

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if not self.is_stride:
            x = self._layout_convert_x(x)

        # last dimension is about irreps, except for it
        outer_shape = x.shape[:-1]
        L_x = self._L_x
        L_out = self._L_true
        size = 2 * L_out + 1

        out = torch.zeros(
            (outer_shape + (size, size)),
            dtype=self.complex_dtype,
            device=x.device,
        )

        x_0 = self._x_to_fourier(x)
        x_f = torch.fft.fft2(
            x_0,
            s=(size, size),
            norm=self.fft_norm
        )

        base = x_f
        c = L_out - L_x
        r = 2 * L_x + 1
        out[..., c:c + r, c:c + r] =\
            out[..., c:c + r, c:c + r] + x_0[..., :r, :r]

        for v in range(2, self.correlation + 1):
            c = L_out - L_x * v
            r = 2 * L_x * v + 1
            base = base * x_f
            out[..., c:c + r, c:c + r] =\
                out[..., c:c + r, c:c + r] +\
                torch.fft.ifft2(base, norm=self.fft_norm)[..., :r, :r]

        x = self._out_to_sph(out)  # [..., (L_out+1)**2]
        x = x * self.path_w

        if not self.is_stride:
            x = self._layout_convert_out.reverse(x)

        data[self.key_x] = x
        return data
