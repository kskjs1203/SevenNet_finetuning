from typing import Callable, List, Tuple

from e3nn.o3 import Irreps

import sevenn._keys as KEY

from .convolution import (
    GauntConvolution,
    HermitianGauntConvolution,
    IrrepsConvolution,
)
from .equivariant_gate import EquivariantGate
from .equivariant_product_basis import EquivariantProductBasis
from .gaunt_product_basis import GauntProductBasis, GauntProductBasisWithWeight
from .linear import IrrepsLinear, IrrepsStrideLinear
from .self_connection import (
    SelfConnectionLinearIntro,
    SelfConnectionStrideLinearIntro,
)


def NequIP_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Callable,
    act_gate: Callable,
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    **conv_kwargs,
):
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    # convolution part, l>lmax is dropped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
        **conv_kwargs,
    )

    # irreps of x increase to gate_irreps_in
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp,
        irreps_for_gate_in,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()
    block[f'{t}_equivariant_gate'] = gate_layer

    return block


def MACE_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    correlation: int,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x=KEY.NODE_FEATURE,
    data_key_weight_input=KEY.EDGE_EMBEDDING,
    parallel=False,
    **conv_kwargs,
):
    # parity should be sph like
    assert all([p == (-1)**l for _, (l, p) in irreps_out])
    block = {}
    sc_intro, sc_outro = self_connection_pair

    feature_mul = irreps_out[0].mul
    # multiplicity should be all same
    assert all([m == feature_mul for m, _ in irreps_out])
    irreps_out_si2 = Irreps()
    for _, ir in irreps_out_tp:
        irreps_out_si2 += Irreps(f'{feature_mul}x{str(ir)}')

    irreps_node_attr = Irreps(f'{num_species}x0e')

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_out,
    )
    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    # convolution part, l>lmax is dropped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
        **conv_kwargs,
    )
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp, irreps_out_si2,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_equivariant_product_basis'] = EquivariantProductBasis(
        irreps_x=irreps_out_si2,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=num_species,
    )
    block[f'{t}_self_interaction_3'] = IrrepsLinear(
        irreps_out, irreps_out,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_self_connection_outro'] = sc_outro()
    return block


def Gaunt_gate_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Callable,
    act_gate: Callable,
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    **conv_kwargs,
):
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    if not (irreps_x.lmax == 0 or irreps_out.lmax == 0):
        # convolution part, l>lmax is dropped as defined in irreps_out
        block[f'{t}_convolution'] = HermitianGauntConvolution(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            data_key_weight_input=data_key_weight_input,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            is_parallel=parallel,
            is_stride=False,
            _compute_edge_fft=(t in [0, 1]),
            _init_edge_fourier=(t in [0, 1]),  # care for test
            **conv_kwargs,
        )
    else:
        # convolution part, l>lmax is dropped as defined in irreps_out
        block[f'{t}_convolution'] = IrrepsConvolution(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out_tp,
            data_key_weight_input=data_key_weight_input,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            is_parallel=parallel,
            **conv_kwargs,
        )

    # irreps of x increase to gate_irreps_in
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp,
        irreps_for_gate_in,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()
    block[f'{t}_equivariant_gate'] = gate_layer

    return block


def Gaunt_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    correlation: int,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    is_stride: bool = True,
    stp_with_weight: bool = True,
    **conv_kwargs,
):
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair
    assert sc_intro == SelfConnectionLinearIntro
    sc_intro = (
        SelfConnectionStrideLinearIntro
        if is_stride
        else SelfConnectionLinearIntro
    )

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_out_tp,
        is_stride=is_stride,
    )

    linear = IrrepsStrideLinear if is_stride else IrrepsLinear
    product_basis = (
        GauntProductBasisWithWeight
        if stp_with_weight
        else GauntProductBasis
    )

    block[f'{t}_self_interaction_1'] = linear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    # convolution part, l>lmax is dropped as defined in irreps_out
    block[f'{t}_convolution'] = GauntConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
        _compute_edge_fft=(t in [0, 1]),
        _init_edge_fourier=(t == 0),
        is_stride=is_stride,
        **conv_kwargs,
    )

    block[f'{t}_self_interaction_2'] = linear(
        irreps_out_tp, irreps_out_tp,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()

    block[f'{t}_gaunt_product_basis'] = product_basis(
        irreps_x=irreps_out_tp,
        irreps_out=irreps_out,
        correlation=correlation,
        fit_gaunt_to_w3j_mode='norm',
        fft_norm='ortho',
        is_stride=is_stride,
    )

    return block
