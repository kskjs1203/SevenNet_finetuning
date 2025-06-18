import os
import pickle

import e3nn.o3
import torch
from e3nn.util.jit import compile_mode

# better to declare as class variable, but this is easier
# as I want to use it with TorchScript

COEFF_DIR = f'{os.path.dirname(__file__)}/coeff/'
_Y_coeffs = None  # read from pickle, should not change
_Z_coeffs = None


def weight_align_matrix(L, with_irrep_weight=False):
    m = torch.eye(L + 1)
    indi = []
    for l in range(L + 1):
        if with_irrep_weight:
            m[l] *= (2 * l + 1) ** 0.5
        for _ in range(2 * l + 1):
            indi.append(l)
    return m[indi].T.contiguous()


@compile_mode('script')
class ToFourierBasis(torch.nn.Module):
    """Convert irreps to corresponding fourier basis

    Attributes
    ----------
    irreps_in : e3nn.o3.Irreps
        target irreps
    irrep_dim : int
        (L+1)**2, dimension of irrep (without multiplicity)
    complex_dtype : torch.dtype
        torch dtype for complex numbers, determined by torch.get_default_dtype()
    yy : dict
        coefficints up to L
    """

    def __init__(self, irreps_in: e3nn.o3.Irreps):
        """
        Parameters
        ----------
        irreps_in : e3nn.o3.Irreps
            target irreps, should be sorted & simplified
        """
        assert irreps_in == irreps_in.sort().irreps.simplify()
        assert all([irreps_in[0].mul == mul for mul, _ in irreps_in])
        super().__init__()
        global _Y_coeffs
        if _Y_coeffs is None:
            y_pkl = os.path.join(COEFF_DIR, 'yy.pkl')
            with open(y_pkl, 'rb') as f:
                _Y_coeffs = pickle.load(f)  # [l][m][v][u]
        L = irreps_in.lmax
        default_dtype = torch.get_default_dtype()
        complex_dtype =\
            torch.complex64 if default_dtype == torch.float32 else torch.complex128
        yy_dim = ((L + 1)**2, 2 * L + 1, 2 * L + 1)
        yy = torch.zeros(size=yy_dim, dtype=complex_dtype)
        for l in range(L + 1):
            for m in range(-l, l + 1):
                for v in _Y_coeffs[l][m]:
                    for u in _Y_coeffs[l][m][v]:
                        yy[l**2 + m + l][u + L][v + L] = _Y_coeffs[l][m][v][u]
        self.irreps_in = irreps_in
        self.irrep_dim = (L + 1)**2
        self.complex_dtype = complex_dtype
        self._uv_shape = yy.shape[1:]
        self.register_buffer('yy', yy.flatten(1, 2))

    def forward(self, irreps_tensor: torch.Tensor) -> torch.Tensor:
        """Return corresponding coefficients in fourier basis

        Parameters
        ----------
        irreps_tensor : torch.Tensor
            irreps tensor with strided layout (..., (L+1)**2)

        Returns
        -------
        torch.Tensor
            corresponding u, v coefficients (..., U, V)
        """
        assert irreps_tensor.shape[-1] == self.irrep_dim
        output_shape = irreps_tensor.shape[:-1] + self._uv_shape
        x = irreps_tensor.reshape((-1, self.irrep_dim)).to(dtype=self.complex_dtype)
        return (x @ self.yy).reshape(output_shape)


@compile_mode('script')
class ToSphericalBasis(torch.nn.Module):
    """Convert coeff tensor of fourierb basis to spherical basis (irreps)

    Attributes
    ----------
    L : int
        max L
    complex_dtype : torch.dtype
        torch dtype for complex numbers, determined by torch.get_default_dtype()
    zz : dict
        coefficints up to L_max
    """

    def __init__(self, L_in: int, L_max: int = -1, hermitian_input=False):
        """
        Parameters
        ----------
        L_in : int
            corresponding L of input fourier tensor
        L_max : int
            if given, resulting irreps will be truncated by this, defaults to L_in
        """
        super().__init__()
        global _Z_coeffs
        if _Z_coeffs is None:
            z_pkl = os.path.join(COEFF_DIR, 'zz.pkl')
            with open(z_pkl, 'rb') as f:
                _Z_coeffs = pickle.load(f)  # [v][u][l][m]
        default_dtype = torch.get_default_dtype()
        complex_dtype =\
            torch.complex64 if default_dtype == torch.float32 else torch.complex128

        L = L_in
        zz_dim = (2 * L + 1, 2 * L + 1, (L + 1)**2)  # lm, u, v
        zz = torch.zeros(size=zz_dim, dtype=complex_dtype)
        for u in range(-L, L + 1):
            for v in range(-L, L + 1):
                for l in _Z_coeffs[v][u]:
                    if l > L:
                        continue
                    for m in _Z_coeffs[v][u][l]:
                        zz[u + L][v + L][l**2 + m + l] = _Z_coeffs[v][u][l][m]
        if L_max != -1:
            assert L_max <= L and L_max >= 0
            zz = zz[:, :, :(L_max + 1)**2]

        if hermitian_input:
            conv_dim = 2 * L_in + 1
            nz = torch.fft.ifftshift(zz, dim=(0, 1))[:, :conv_dim // 2 + 1, :]
            cnt = 0
            for l in range(L_max + 1):
                for m in range(-l, l + 1):
                    if m != 0:
                        nz[:, :, cnt] *= 2
                    cnt += 1
            zz = nz

        self.L = L_max
        self.complex_dtype = complex_dtype

        self._uv_shape = zz.shape[:2]
        self._uv_numel = self._uv_shape.numel()
        self.register_buffer('zz', zz.flatten(0, 1))
        self._ir_dim = (L_max + 1)**2

    def forward(self, fourier_tensor: torch.Tensor) -> torch.Tensor:
        """Return corresponding coefficients in spherical harmonics basis

        Parameters
        ----------
        fourier_tensor : torch.Tensor
            tensor representing coeffs of 2D fourier basis (..., U, V)

        Returns
        -------
        torch.Tensor
            corresponding coeffs in spherical harmonics basis (irreps)
        """
        assert fourier_tensor.shape[-2:] == self._uv_shape  # last two dim
        output_shape = fourier_tensor.shape[:-2] + (self._ir_dim,)
        x = fourier_tensor.reshape((-1,) + (self._uv_numel,))
        return (x @ self.zz).real.reshape(output_shape)


def fit_gaunt_to_w3j(L1, L2, mode='norm'):
    """
    Based on https://en.wikipedia.org/wiki/3-j_symbol,
    return ratio between CGTP and GTP for scalining.
    Basically, GTP_computed = prefactor * w3j_computed
    """
    assert mode in ['norm']
    pi = torch.pi
    Lmax = L1 + L2
    ratio = [[] for _ in range(Lmax + 1)]
    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            for lout in range(abs(l1 - l2), l1 + l2 + 1):
                w3j = e3nn.o3.wigner_3j(l1, l2, lout)[l1][l2][lout]
                _l_mult = (2 * l1 + 1) * (2 * l2 + 1) * (2 * lout + 1)
                ratio[lout].append((_l_mult / (4 * pi))**0.5 * w3j)
    out = torch.ones(Lmax + 1)
    for i, prefactors in enumerate(ratio):
        if mode == 'norm':
            out[i] = 1 / torch.norm(torch.stack(prefactors))
        else:
            raise ValueError(f'Unknown mode: {mode}')
    return out


def init_edge_rot_mat(edge_distance_vec):
    """
    Copied from EquiformerV2
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            'Error edge_vec_0_distance: {}'.format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copies of the random vectors
    # in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors,
    # at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()


class _rfft2(torch.autograd.Function):
    """
    Pytorch implementation of backward _rfft2 is slow.
    to grad as output of c2c fft, requiring:
    elementwise preprocessing steps + full fft

    This implementation of backward path is much faster
    https://github.com/locuslab/pytorch_fft

    Keeps torch's irfft2 intact, as it is not bad as rfft2
    (They are not compared, but it uses r2c fft, at least)
    """

    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        ctx.n2 = x.shape[-1]
        ctx.n1 = x.shape[-2]
        return torch.fft.rfft2(x)

    @staticmethod
    def backward(ctx, *grad_outputs) -> torch.Tensor:
        dy, = grad_outputs
        if ctx.n2 & 1:
            dy[..., 1:] /= 2
        else:
            dy[..., 1:-1] /= 2
        # TODO: somehow, there is large D2D memcpy occurs
        # it is very strange as other irfft2 ops (gaunt convolution),
        # does not raise such D2D memcpy.
        # Unfortunately, I failed to figure it out...
        return torch.fft.irfft2(dy, s=(ctx.n1, ctx.n2), norm='forward')


def rfft2(x):
    return _rfft2.apply(x)
