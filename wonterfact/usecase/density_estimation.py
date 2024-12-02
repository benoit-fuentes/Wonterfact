from __future__ import annotations

import abc
import typing
from dataclasses import InitVar, dataclass, field

import numpy as np
import numpy.random as npr
import numpy.typing as npt
import scipy.stats as scs
from tqdm import tqdm

import wonterfact as wtf

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

    from wonterfact._core_nodes import DynNodeData

param_name = typing.Literal["sign", "coef", "zoom_out"]
rng = npr.default_rng()


def logmeanexp(vector):
    max_val = np.max(vector)

    log_sum_exp = np.log(np.sum(np.exp(vector - max_val)))

    return log_sum_exp + max_val - np.log(vector.size)


@dataclass
class DecoderParam:
    value: npt.NDArray
    prior: npt.NDArray


@dataclass
class Layer(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return type"""

    def repr(self):
        return self.name


@dataclass
class DataLayer(Layer):
    size_out: int
    name = "data"

    def compile(self, data: npt.NDArray):
        if data.ndim != 3 or data.shape[-1] != 2:
            msg = "Shape of data array should be (batch_size, data_dimension, 2)"
            raise ValueError(msg)
        if data.shape[1] != self.size_out:
            msg = "Wrong dimension for data"
            raise ValueError(msg)

        root = wtf.Root(
            # inference_mode="EM",
            inference_mode="VB-MCMC",
            cost_computation_iter=0,
            update_type="regular",
            # update_type="parabolic",
            acceleration_start_iter=3,
            verbose_iter=0,
            stop_estim_threshold=0,
        )
        niter_until_maxdrawings = 1
        obs_ngz = wtf.PosObserver(
            tensor=data,
            index_id="ngz",
            norm_axis=(2,),
            name=self.name,
            drawings_step=data.sum() / niter_until_maxdrawings,
        )
        obs_ngz.new_child(root)
        return root, obs_ngz


@dataclass
class CodeLayer(Layer):
    size_in: int
    code_prior: npt.NDArray = field(init=False)

    _code_prior: InitVar[npt.NDArray | None] = None
    name = "code"

    def __post_init__(self, _code_prior: npt.NDArray | None = None):
        if _code_prior is None:
            self.code_prior = np.ones((self.size_in, 2))
        else:
            if _code_prior.shape != (self.size_in, 2):
                msg = "Wrong shape for  `_code_prior`"
                raise ValueError(msg)
            self.code_prior = _code_prior

    def compile(self, batch_size: int) -> wtf.LeafDirichlet:
        tensor = np.ones((batch_size, self.size_in, 2))
        tensor[:, ...] = self.code_prior
        # tensor[:, ...] = self.code_prior + rng.uniform(size=tensor.shape) * 0.1
        # tensor[:, :, ...] = np.array([1, 0])
        tensor = tensor / tensor.sum(-1, keepdims=True)
        return wtf.LeafDirichlet(
            index_id="ngz",
            norm_axis=(2,),
            tensor=tensor,
            prior_shape=self.code_prior,
            name=self.name,
        )

    def update_from_tree(self, tree: wtf.Root, update_prior: bool):
        if not update_prior:
            return self
        try:
            code_prior = tree.get_leaf(self.name).shape_parent.tensor_as_numpy
        except ValueError as err:
            msg = "Code layer does not exist"
            raise ValueError(msg) from err
        return CodeLayer(size_in=self.size_in, _code_prior=code_prior)


@dataclass
class DipoleLayer(Layer):
    size_in: int
    size_out: int
    uid: Hashable

    @abc.abstractmethod
    def compile(self, update_decoder: bool, batch_size: int) -> tuple[DynNodeData, DynNodeData]:
        """Compile a layer as a piece on wonterfact tree model"""

    @abc.abstractmethod
    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool) -> DipoleLayer:
        """Return a copy with updated parameter priors and values"""


@dataclass
class ShrinkLayer(DipoleLayer):
    shrink: DecoderParam
    name = "shrink"

    def compile(self, update_decoder: bool, batch_size: int):  # noqa: ARG002
        shrinking = wtf.LeafDirichlet(
            index_id="gzv",
            norm_axis=(2,),
            name=f"shrink_{self.uid}",
            tensor=self.shrink.value,
            prior_shape=self.shrink.prior,
            update_period=int(update_decoder),
        )
        towards_code = wtf.Multiplier(index_id="ngv", name=f"code_{self.uid}")
        towards_code.new_parent(shrinking)
        towards_data = wtf.Proxy(index_id="ngz")
        towards_code.new_child(towards_data, index_id_for_child="ngz")
        return towards_data, towards_code

    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool):
        try:
            shrink_leaf = tree.get_leaf(f"shrink_{self.uid}")
        except ValueError as err:
            msg = "Requested shrink layer does not exist"
            raise ValueError(msg) from err
        if snapshot:
            raise NotImplementedError
        shrink_posterior = shrink_leaf.shape_parent.tensor_as_numpy
        shrink_val = shrink_posterior / shrink_posterior.sum(shrink_leaf.norm_axis, keepdims=True)
        return ShrinkLayer(
            uid=self.uid,
            size_in=self.size_in,
            size_out=self.size_out,
            shrink=DecoderParam(
                value=shrink_val, prior=shrink_posterior if update_prior else self.shrink.prior
            ),
        )


@dataclass
class NoiseLayer(DipoleLayer):
    mix: DecoderParam = field(init=False)
    noise_prior: npt.NDArray = field(init=False)
    _mix: InitVar[DecoderParam | None] = None
    _noise_prior: InitVar[npt.NDArray | None] = None
    name = "noise"

    def __post_init__(
        self, _mix: DecoderParam | None = None, _noise_prior: npt.NDArray | None = None
    ):
        if self.size_in != self.size_out:
            msg = "size_in and size_out must be equals for noise layers"
            raise AttributeError(msg)
        if _mix is None:
            self.mix = DecoderParam(prior=np.ones(2), value=np.ones(2) / 2)
        else:
            if not _mix.prior.shape == _mix.value.shape == (2,):
                msg = "Wrong shape for _mix"
                raise ValueError(msg)
            self.mix = _mix
        if _noise_prior is None:
            self.noise_prior = np.ones((self.size_in, 2)) * 100
        else:
            if _noise_prior.shape != ((self.size_in, 2)):
                msg = "Wrong shape for _noise_prior"
                raise ValueError(msg)
            self.noise_prior = _noise_prior

    def compile(self, update_decoder: bool, batch_size: int):
        update_period = 1 if update_decoder else 0
        towards_code = wtf.Multiplexer(index_id="mngz", name=f"plug_here_{self.uid}")
        noise_leaf = wtf.LeafDirichlet(
            index_id="ngz",
            norm_axis=(2,),
            tensor=np.ones((batch_size, self.size_out, 2)) / 2,
            prior_shape=self.noise_prior,
            name=f"noise_{self.uid}",
        )
        towards_code.new_parent(noise_leaf)
        mix_leaf = wtf.LeafDirichlet(
            index_id="m",
            tensor=self.mix.value,
            name=f"mix_{self.uid}",
            prior_shape=self.mix.prior,
            update_period=update_period,
        )
        toward_data = wtf.Multiplier(index_id="ngz", name=f"code_{self.uid}")
        toward_data.new_parents(towards_code, mix_leaf)
        return toward_data, towards_code

    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool):
        try:
            mix_leaf = tree.get_leaf(f"mix_{self.uid}")
            # noise_leaf = tree.get_leaf(f"noise_{self.uid}")
        except ValueError as err:
            msg = "Requested noise layer does not exist"
            raise ValueError(msg) from err
        if snapshot and not update_prior:
            return NoiseLayer(
                uid=self.uid,
                size_in=self.size_in,
                size_out=self.size_out,
                _mix=DecoderParam(value=mix_leaf.tensor_as_numpy, prior=self.mix.prior),
                _noise_prior=self.noise_prior,
            )
        raise NotImplementedError


@dataclass
class DenseLayer(DipoleLayer):
    sign: DecoderParam = field(init=False)
    covar: DecoderParam = field(init=False)
    shrink: DecoderParam = field(init=False)

    _sign: InitVar[DecoderParam | None] = None
    _covar: InitVar[DecoderParam | None] = None
    _shrink: InitVar[DecoderParam | None] = None

    name = "dense"

    def __post_init__(
        self, _sign: DecoderParam | None, _covar: DecoderParam | None, _shrink: DecoderParam | None
    ):
        if _sign is None:
            sign_prior = np.zeros((self.size_in, self.size_out, 2)) + 0.6 + np.array([1, 0]) * 0
            # sign_prior = np.zeros((self.size_in, self.size_out, 2)) + 1 + np.array([1, 0]) * 0
            # sign_val = sign_prior / sign_prior.sum(2, keepdims=True)
            sign_val = np.zeros((self.size_in, self.size_out, 2)) + np.array([1, 0]) * 100
            sign_val = sign_val / sign_val.sum(2, keepdims=True)
            self.sign = DecoderParam(prior=sign_prior, value=sign_val)
        else:
            if not _sign.prior.shape == _sign.value.shape == (self.size_in, self.size_out, 2):
                msg = "wrong shape for `_sign` argument"
                raise ValueError(msg)
            self.sign = _sign
        if _covar is None:
            covar_prior = np.ones((self.size_in, self.size_out))
            min_size = min(self.size_in, self.size_out)
            covar_prior[:min_size, :min_size] += np.eye(min_size) * 0
            # covar_value = covar_prior / covar_prior.sum(1, keepdims=True)
            covar_value = np.ones((self.size_in, self.size_out))
            min_size = min(self.size_in, self.size_out)
            covar_value[:min_size, :min_size] += np.eye(min_size) * 100
            covar_value = covar_value / covar_value.sum(1, keepdims=True)
            self.covar = DecoderParam(prior=covar_prior, value=covar_value)
        else:
            if not _covar.prior.shape == _covar.value.shape == (self.size_in, self.size_out):
                msg = "wrong shape for `_covar` argument"
                raise ValueError(msg)
            self.covar = _covar
        if _shrink is None:
            shrink_prior = np.ones((self.size_in, 2, 2)) + np.eye(2) * 10
            # shrink_prior = np.ones((self.size_in, 2, 2)) + np.eye(2) * 0
            # shrink_value = shrink_prior / shrink_prior.sum(2, keepdims=True)
            shrink_value = np.ones((self.size_in, 2, 2)) + np.eye(2) * 1000
            shrink_value = shrink_value / shrink_value.sum(2, keepdims=True)
            self.shrink = DecoderParam(prior=shrink_prior, value=shrink_value)
        else:
            if not _shrink.value.shape == _shrink.prior.shape == (self.size_in, 2, 2):
                msg = "wrong shape for `_shrink` argument"
                raise ValueError(msg)
            self.shrink = _shrink

    def compile(self, update_decoder: bool, batch_size: int):  # noqa: ARG002
        update_period = 1 if update_decoder else 0
        sign = wtf.LeafDirichlet(
            index_id="fgs",
            norm_axis=(2,),
            name=f"sign_{self.uid}",
            tensor=self.sign.value,
            prior_shape=self.sign.prior,
            update_period=update_period,
        )
        covar = wtf.LeafDirichlet(
            index_id="fg",
            norm_axis=(1,),
            name=f"covar_{self.uid}",
            tensor=self.covar.value,
            prior_shape=self.covar.prior,
            update_period=update_period,
        )
        signed_covar = wtf.Multiplier(
            index_id="fgs", name=f"signed_covar_{self.uid}", update_period=update_period
        )
        signed_covar.new_parents(sign, covar)
        towards_code = wtf.Multiplier(index_id="nfzs", name=f"plug_here_{self.uid}")
        towards_code.new_parent(signed_covar)
        sign_flipper = wtf.LeafDirichlet(
            index_id="zsw",
            norm_axis=(2,),
            tensor=np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            update_period=0,
            name=f"flipper_{self.uid}",
        )
        after_rotation = wtf.Multiplier(index_id="nfw", name=f"rotated_code_{self.uid}")
        after_rotation.new_parents(towards_code, sign_flipper)
        zoom_out = wtf.LeafDirichlet(
            index_id="fwz",
            norm_axis=(2,),
            name=f"shrink_{self.uid}",
            tensor=self.shrink.value,
            prior_shape=self.shrink.prior,
            update_period=update_period,
        )
        after_zoom_out = wtf.Multiplier(index_id="nfz", name=f"zoom_out_code_{self.uid}")
        after_zoom_out.new_parents(zoom_out, after_rotation)
        towards_data = wtf.Smoothstep(index_id="ngz", name=f"code_{self.uid}")
        after_zoom_out.new_child(towards_data, index_id_for_child="ngz")
        return towards_data, towards_code

    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool):
        try:
            sign_leaf = tree.get_leaf(f"sign_{self.uid}")
            covar_leaf = tree.get_leaf(f"covar_{self.uid}")
            shrink_leaf = tree.get_leaf(f"shrink_{self.uid}")
        except ValueError as err:
            msg = "Requested dense layer does not exist"
            raise ValueError(msg) from err
        if snapshot:
            if update_prior:
                raise NotImplementedError
            return DenseLayer(
                uid=self.uid,
                size_in=self.size_in,
                size_out=self.size_out,
                _sign=DecoderParam(value=sign_leaf.tensor_as_numpy, prior=self.sign.prior),
                _covar=DecoderParam(value=covar_leaf.tensor_as_numpy, prior=self.covar.prior),
                _shrink=DecoderParam(value=shrink_leaf.tensor_as_numpy, prior=self.shrink.prior),
            )
        sign_posterior = sign_leaf.shape_parent.tensor_as_numpy
        sign_val = sign_posterior / sign_posterior.sum(sign_leaf.norm_axis, keepdims=True)
        covar_posterior = covar_leaf.shape_parent.tensor_as_numpy
        covar_val = covar_posterior / covar_posterior.sum(covar_leaf.norm_axis, keepdims=True)
        shrink_posterior = shrink_leaf.shape_parent.tensor_as_numpy
        shrink_val = shrink_posterior / shrink_posterior.sum(shrink_leaf.norm_axis, keepdims=True)
        return DenseLayer(
            uid=self.uid,
            size_in=self.size_in,
            size_out=self.size_out,
            _sign=DecoderParam(
                value=sign_val, prior=sign_posterior if update_prior else self.sign.prior
            ),
            _covar=DecoderParam(
                value=covar_val, prior=covar_posterior if update_prior else self.covar.prior
            ),
            _shrink=DecoderParam(
                value=shrink_val, prior=shrink_posterior if update_prior else self.shrink.prior
            ),
        )


@dataclass
class DenseLayer2(DipoleLayer):
    sign: DecoderParam = field(init=False)
    covar: DecoderParam = field(init=False)
    center: DecoderParam = field(init=False)
    mix: DecoderParam = field(init=False)

    _sign: InitVar[DecoderParam | None] = None
    _covar: InitVar[DecoderParam | None] = None
    _center: InitVar[DecoderParam | None] = None
    _mix: InitVar[DecoderParam | None] = None

    name = "dense2"

    def __post_init__(
        self,
        _sign: DecoderParam | None,
        _covar: DecoderParam | None,
        _center: DecoderParam | None,
        _mix: DecoderParam | None,
    ):
        if _sign is None:
            sign_prior = np.zeros((self.size_in, self.size_out, 2)) + 0.6 + np.array([1, 0]) * 0
            sign_val = sign_prior / sign_prior.sum(2, keepdims=True)
            self.sign = DecoderParam(prior=sign_prior, value=sign_val)
        else:
            if not _sign.prior.shape == _sign.value.shape == (self.size_in, self.size_out, 2):
                msg = "wrong shape for `_sign` argument"
                raise ValueError(msg)
            self.sign = _sign
        if _covar is None:
            covar_prior = np.ones((self.size_in, self.size_out))
            min_size = min(self.size_in, self.size_out)
            covar_prior[:min_size, :min_size] += np.eye(min_size) * 0
            covar_value = covar_prior / covar_prior.sum(1, keepdims=True)
            self.covar = DecoderParam(prior=covar_prior, value=covar_value)
        else:
            if not _covar.prior.shape == _covar.value.shape == (self.size_in, self.size_out):
                msg = "wrong shape for `_covar` argument"
                raise ValueError(msg)
            self.covar = _covar
        if _center is None:
            center_prior = np.ones((self.size_in, 2))
            self.center = DecoderParam(prior=center_prior, value=center_prior / 2)
        else:
            if not _center.value.shape == _center.prior.shape == (self.size_in, 2):
                msg = "wrong shape for `_shrink` argument"
                raise ValueError(msg)
            self.center = _center
        if _mix is None:
            mix_prior = np.ones((self.size_in, 2))
            self.mix = DecoderParam(prior=mix_prior, value=mix_prior / 2)
        else:
            if not _mix.value.shape == _mix.prior.shape == (self.size_in, 2):
                msg = "wrong shape for `_shrink` argument"
                raise ValueError(msg)
            self.mix = _mix

    def compile(self, update_decoder: bool, batch_size: int):
        update_period = 1 if update_decoder else 0
        sign = wtf.LeafDirichlet(
            index_id="fgs",
            norm_axis=(2,),
            name=f"sign_{self.uid}",
            tensor=self.sign.value,
            prior_shape=self.sign.prior,
            update_period=update_period,
        )
        covar = wtf.LeafDirichlet(
            index_id="fg",
            norm_axis=(1,),
            name=f"covar_{self.uid}",
            tensor=self.covar.value,
            prior_shape=self.covar.prior,
            update_period=update_period,
        )
        signed_covar = wtf.Multiplier(
            index_id="fgs", name=f"signed_covar_{self.uid}", update_period=update_period
        )
        signed_covar.new_parents(sign, covar)
        towards_code = wtf.Multiplier(index_id="nfzs", name=f"plug_here_{self.uid}")
        towards_code.new_parent(signed_covar)
        sign_flipper = wtf.LeafDirichlet(
            index_id="zsw",
            norm_axis=(2,),
            tensor=np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            update_period=0,
            name=f"flipper_{self.uid}",
        )
        after_rotation = wtf.Multiplier(index_id="nfw", name=f"rotated_code_{self.uid}")
        after_rotation.new_parents(towards_code, sign_flipper)
        center = wtf.LeafDirichlet(
            index_id="fw",
            norm_axis=(1,),
            name=f"center_{self.uid}",
            tensor=self.center.value,
            prior_shape=self.center.prior,
            update_period=update_period,
        )
        duplicate = wtf.LeafDirichlet(
            index_id="n",
            norm_axis=(),
            name=f"duplicate_{self.uid}",
            tensor=np.ones(batch_size),
            update_period=0,
        )
        center2 = wtf.Multiplier(index_id="nfw", name=f"center2_{self.uid}")
        center2.new_parents(center, duplicate)
        premix = wtf.Multiplexer(index_id="cnfw", name=f"premix_{self.uid}")
        premix.new_parents(after_rotation, center2)
        mix = wtf.LeafDirichlet(
            index_id="fc",
            norm_axis=(1,),
            name=f"mix_{self.uid}",
            tensor=self.mix.value,
            prior_shape=self.mix.prior,
            update_period=update_period,
        )

        after_zoom_out = wtf.Multiplier(index_id="nfw", name=f"shrink_code_{self.uid}")
        after_zoom_out.new_parents(premix, mix)
        towards_data = wtf.Smoothstep(index_id="ngz", name=f"code_{self.uid}")
        after_zoom_out.new_child(towards_data, index_id_for_child="ngz")
        return towards_data, towards_code

    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool):
        try:
            sign_leaf = tree.get_leaf(f"sign_{self.uid}")
            covar_leaf = tree.get_leaf(f"covar_{self.uid}")
            center_leaf = tree.get_leaf(f"center_{self.uid}")
            mix_leaf = tree.get_leaf(f"mix_{self.uid}")
        except ValueError as err:
            msg = "Requested dense layer does not exist"
            raise ValueError(msg) from err
        if snapshot:
            if update_prior:
                raise NotImplementedError
            return DenseLayer2(
                uid=self.uid,
                size_in=self.size_in,
                size_out=self.size_out,
                _sign=DecoderParam(value=sign_leaf.tensor_as_numpy, prior=self.sign.prior),
                _covar=DecoderParam(value=covar_leaf.tensor_as_numpy, prior=self.covar.prior),
                _center=DecoderParam(value=center_leaf.tensor_as_numpy, prior=self.center.prior),
                _mix=DecoderParam(value=mix_leaf.tensor_as_numpy, prior=self.mix.prior),
            )
        sign_posterior = sign_leaf.shape_parent.tensor_as_numpy
        sign_val = sign_posterior / sign_posterior.sum(sign_leaf.norm_axis, keepdims=True)
        covar_posterior = covar_leaf.shape_parent.tensor_as_numpy
        covar_val = covar_posterior / covar_posterior.sum(covar_leaf.norm_axis, keepdims=True)
        center_posterior = center_leaf.shape_parent.tensor_as_numpy
        center_val = center_posterior / center_posterior.sum(center_leaf.norm_axis, keepdims=True)
        mix_posterior = mix_leaf.shape_parent.tensor_as_numpy
        mix_val = mix_posterior / mix_posterior.sum(mix_leaf.norm_axis, keepdims=True)
        return DenseLayer2(
            uid=self.uid,
            size_in=self.size_in,
            size_out=self.size_out,
            _sign=DecoderParam(
                value=sign_val, prior=sign_posterior if update_prior else self.sign.prior
            ),
            _covar=DecoderParam(
                value=covar_val, prior=covar_posterior if update_prior else self.covar.prior
            ),
            _center=DecoderParam(
                value=center_val, prior=center_posterior if update_prior else self.center.prior
            ),
            _mix=DecoderParam(
                value=mix_val, prior=mix_posterior if update_prior else self.mix.prior
            ),
        )


@dataclass
class ShortCutLayer(DipoleLayer):
    jump: int
    mix: DecoderParam = field(init=False)

    _mix: InitVar[DecoderParam | None] = None
    name = "shortcut"

    def __post_init__(self, _mix: DecoderParam | None = None):
        if self.size_in != self.size_out:
            msg = "size in and size out should be equals"
            raise NotImplementedError(msg)
        if _mix is None:
            self.mix = DecoderParam(value=np.ones(2) / 2, prior=np.ones(2))
        else:
            if not _mix.value.shape == _mix.prior.shape == (2,):
                msg = "wrong dimension for _mix"
                raise ValueError(msg)
            self.mix = _mix

    def compile(self, update_decoder, batch_size):  # noqa: ARG002
        update_period = 1 if update_decoder else 0
        towards_code = wtf.Multiplexer(index_id="mngz", name=f"plug_here_{self.uid}")
        mix_leaf = wtf.LeafDirichlet(
            index_id="m",
            tensor=self.mix.value,
            name=f"mix_{self.uid}",
            prior_shape=self.mix.prior,
            update_period=update_period,
        )
        toward_data = wtf.Multiplier(index_id="ngz", name=f"code_{self.uid}")
        toward_data.new_parents(towards_code, mix_leaf)
        return toward_data, towards_code

    def update_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool):
        try:
            mix_leaf = tree.get_leaf(f"mix_{self.uid}")
        except ValueError as err:
            msg = "Requested noise layer does not exist"
            raise ValueError(msg) from err
        if snapshot:
            if update_prior:
                raise NotImplementedError
            return ShortCutLayer(
                uid=self.uid,
                size_in=self.size_in,
                size_out=self.size_out,
                jump=self.jump,
                _mix=DecoderParam(value=mix_leaf.tensor_as_numpy, prior=self.mix.prior),
            )
        mix_posterior = mix_leaf.shape_parent.tensor_as_numpy
        mix_val = mix_posterior / mix_posterior.sum(mix_leaf.norm_axis, keepdims=True)
        return ShortCutLayer(
            uid=self.uid,
            size_in=self.size_in,
            size_out=self.size_out,
            jump=self.jump,
            _mix=DecoderParam(
                value=mix_val, prior=mix_posterior if update_prior else self.mix.prior
            ),
        )

    def repr(self):
        return f"{self.name}(+{self.jump})"


def toy_model(code_prior: npt.NDArray, shrink_prior: npt.NDArray):
    mymodel = MyModel(
        data_layer=DataLayer(size_out=1),
        layers=[
            ShrinkLayer(
                size_in=1,
                size_out=1,
                uid="a",
                shrink=DecoderParam(
                    prior=shrink_prior, value=shrink_prior / shrink_prior.sum(2, keepdims=True)
                ),
            )
        ],
        code_layer=CodeLayer(size_in=1, _code_prior=code_prior),
    )
    mymodel.layers.append(
        ShrinkLayer(
            size_in=1,
            size_out=1,
            uid="a",
            shrink=DecoderParam(
                prior=shrink_prior, value=shrink_prior / shrink_prior.sum(2, keepdims=True)
            ),
        )
    )
    return mymodel


@dataclass
class MyModel:
    data_layer: DataLayer | None = None
    layers: list[DipoleLayer] = field(default_factory=list)
    shortcuts: list[ShortCutLayer] = field(default_factory=list)
    code_layer: CodeLayer | None = None

    def _check_consistency(self):
        size_to_data = self.data_layer.size_out
        for layer in self.layers:
            if layer.size_in != size_to_data:
                msg = (
                    f"Attibute `size_in` of layer {layer.uid } does not match with the previous one"
                )
                raise ValueError(msg)
            size_to_data = layer.size_out
        if self.code_layer.size_in != size_to_data:
            msg = "Code size does not match with previous layer"
            raise ValueError(msg)

    @property
    def size_out(self) -> int:
        if self.layers:
            return self.layers[-1].size_out
        if self.data_layer is not None:
            return self.data_layer.size_out
        msg = "Undefined property, please add a first data layer"
        raise AttributeError(msg)

    def add_layer(self, layer: Layer):
        if isinstance(layer, DipoleLayer):
            self.layers.append(layer)
        if isinstance(layer, CodeLayer):
            if self.code_layer:
                msg = "code layer already exists"
                raise ValueError(msg)
            self.code_layer = layer
        if isinstance(layer, DataLayer):
            if self.data_layer:
                msg = "data layer already exists"
                raise ValueError(msg)
            self.data_layer = layer

    def make_tree(self, data: npt.NDArray, update_decoder: bool):
        if not self.data_layer or not self.code_layer:
            msg = "Please define a data and code layers"
            raise AttributeError(msg)

        batch_size = data.shape[0]

        # first we compile all the tree pieces
        root, obs_node = self.data_layer.compile(data)
        code_node = self.code_layer.compile(batch_size)
        # if not update_decoder:
        #     code_node.inertia = 0.8
        #     code_node.brake = 0
        input_output = [layer.compile(update_decoder, batch_size) for layer in self.layers]

        # then we link them together
        n_layers = len(self.layers)
        for ii in range(n_layers + 1):
            layer1: DataLayer | DipoleLayer = self.data_layer if ii == 0 else self.layers[ii - 1]
            layer2: CodeLayer | DipoleLayer = self.code_layer if ii == n_layers else self.layers[ii]
            node1 = obs_node if ii == 0 else input_output[ii - 1][1]
            node2 = code_node if ii == n_layers else input_output[ii][0]
            if layer1.size_out != layer2.size_in:
                msg = f"Attibute `size_in` of layer n°{ii} does not match with the previous one"
                raise ValueError(msg)
            node2.new_child(node1)
            if isinstance(layer1, ShortCutLayer):
                if ii + layer1.jump > n_layers + 1:
                    msg = "No layer to plug into for shorcut layer"
                    raise AttributeError(msg)
                layer_to_plug: CodeLayer | DipoleLayer
                if ii + layer1.jump <= n_layers:
                    layer_to_plug = self.layers[ii - 1 + layer1.jump]
                    node_to_plug = input_output[ii - 1 + layer1.jump][0]
                else:
                    layer_to_plug = self.code_layer
                    node_to_plug = code_node
                if layer1.size_out != layer_to_plug.size_in:
                    msg = "Impossible to pair layer n° {} to shorcut: dimensions dismatch"
                    raise AttributeError(msg)
                node_to_plug.new_child(node1)

        return root

    def __repr__(self) -> str:
        str_repr = type(self).__name__ + "(\n"
        if self.data_layer:
            str_repr += f"    [data layer: {self.data_layer.size_out}],\n"
        for layer in self.layers:
            str_repr += (
                f"    [{layer.repr()} layer {layer.uid}: {layer.size_in} -> {layer.size_out}],\n"
            )
        if self.code_layer:
            str_repr += f"    [code layer: {self.code_layer.size_in}]"
        str_repr += "\n)"
        return str_repr

    def run_vbmcmc(
        self,
        data: npt.NDArray,
        update_decoder: bool = True,
        n_burnin: int = 1000,
        n_mcmc: int = 10000,
        callback: typing.Callable[[wtf.Root], None] | None = None,
        show_tqdm: bool = False,
    ):
        tree = self.make_tree(data, update_decoder=update_decoder)
        tree.estimate_param(n_burnin, callback=callback, show_tqdm=show_tqdm)
        tree.tree_traversal("reset_sufficient_statistic", "bottom-up")
        tree.estimate_param(n_mcmc, callback=callback, show_tqdm=show_tqdm)
        return tree

    def _model_from_tree(self, tree: wtf.Root, update_prior: bool, snapshot: bool = False):
        code_layer = (
            self.code_layer.update_from_tree(tree, update_prior) if self.code_layer else None
        )
        new_layers = [layer.update_from_tree(tree, update_prior, snapshot) for layer in self.layers]
        return MyModel(data_layer=self.data_layer, layers=new_layers, code_layer=code_layer)

    def prelearn(
        self,
        dataset: npt.NDArray,
        learning_rate: float,
        epochs: int = 500,
        burning: int = 1000,
        show_tqdm=False,
    ):
        tree = self.run_vbmcmc(
            dataset, update_decoder=True, n_burnin=burning, n_mcmc=epochs, show_tqdm=show_tqdm
        )
        tree.estimate_hyperparam(1000, learning_rate=learning_rate)
        return self._model_from_tree(tree, update_prior=True)

    def fit_decoder(
        self, dataset: npt.NDArray, epochs: int = 500, burning: int = 1000, show_tqdm: bool = False
    ):
        callback = None

        tree = self.run_vbmcmc(
            dataset,
            update_decoder=True,
            n_burnin=burning,
            n_mcmc=epochs,
            callback=callback,
            show_tqdm=show_tqdm,
        )
        # Compute decoder posterior
        tree.get_leaf("code").shape_parent.update_period = 0
        tree.estimate_hyperparam(1000, show_tqdm=show_tqdm)

        return self._model_from_tree(tree, update_prior=False)

    def fit_code_prior(
        self,
        dataset: npt.NDArray,
        code_prior_learning_rate: float = 1.0,
        burning: int = 100,
        epochs: int = 200,
        show_tqdm: bool = False,
    ):
        tree = self.run_vbmcmc(
            dataset, update_decoder=False, n_burnin=burning, n_mcmc=epochs, show_tqdm=show_tqdm
        )
        tree.estimate_hyperparam(1000, learning_rate=code_prior_learning_rate, show_tqdm=show_tqdm)
        new_code_layer = (
            self.code_layer.update_from_tree(tree, update_prior=True) if self.code_layer else None
        )
        return MyModel(data_layer=self.data_layer, layers=self.layers, code_layer=new_code_layer)

    def estimate_log_evidence(self, datapoint: npt.NDArray, n_is: int = 2000):
        assert self.code_layer
        assert self.data_layer
        tree = self.run_vbmcmc(datapoint[None, ...], update_decoder=False, n_burnin=100, n_mcmc=500)
        # code posterior can be very peacky when only one observation: we need many iterations
        tree.estimate_hyperparam(1000)

        # get define code prior and posterior
        code_posterior = tree.get_leaf(self.code_layer.name).shape_parent.tensor_as_numpy

        # remake tree with prior
        tree = self.make_tree(datapoint[None, ...], update_decoder=False)
        tree.estimate_param(1)
        var_list = [0.25, 0.5, 0.75, 1, 1.5]
        # var_list = [0.25, 0.5, 1.0]
        q_proba_tab = [
            [scs.dirichlet(code_dim_shape * var) for code_dim_shape in code_posterior]
            for var in var_list
        ]
        l_arr = np.zeros((len(var_list), n_is))
        for qq, q_proba_list in tqdm(enumerate(q_proba_tab), total=len(q_proba_tab), disable=True):
            code_to_test = np.empty((n_is, self.code_layer.size_in, 2))
            for dim, q_proba in enumerate(q_proba_list):
                code_to_test[:, dim, :] = q_proba.rvs(size=n_is)
            for ii, code in enumerate(code_to_test):
                tree.get_leaf("code").tensor[...] = code
                tree.tree_traversal(
                    "_update_tensor",
                    mode="top-down",
                    method_input=((), {"update_type": "no_update_for_leaves"}),
                    iteration_number=1,
                    type_filter_list=[
                        wtf.BudShape,
                    ],
                )
                l_arr[qq, ii] = -tree.get_cost_func() - logmeanexp(
                    np.array(
                        [
                            qq_proba.logpdf(code_dim)
                            for q_proba_list in q_proba_tab
                            for qq_proba, code_dim in zip(q_proba_list, code)
                        ]
                    )
                )
        return logmeanexp(l_arr.ravel())

    def simulate_data(self, n_data: int, multinomial_drawings: int = 1000):
        assert self.data_layer
        assert self.code_layer
        fake_data = np.zeros((n_data, self.data_layer.size_out, 2))
        tree = self.make_tree(fake_data, update_decoder=False)
        tree.estimate_param(1)
        tree.simulate_data(multinomial_drawings=multinomial_drawings)
        return tree.nodes_by_id["data"].tensor

    def fit_decoder2(
        self, dataset: npt.NDArray, epochs: int = 40, batch_size: int = 40, iter_per_batch: int = 10
    ):
        dataset_size = dataset.shape[0]
        minibatch = dataset[:batch_size]
        tree = self.make_tree(minibatch, update_decoder=False)
        tree.estimate_param(1)
        decoder_nodes = [
            node
            for node in tree.census()
            if isinstance(node, wtf.LeafDirichlet) and "code" not in node.name
        ]
        code_nodes = [
            node
            for node in tree.census()
            if isinstance(node, wtf.LeafDirichlet) and "code" in node.name
        ]
        for __ in tqdm(range(epochs)):
            for n_batch in range(dataset_size // batch_size):
                tree.first_parent.tensor[...] = dataset[
                    n_batch * (batch_size) : (n_batch + 1) * batch_size
                ]
                # first we encode the data using current decodeur
                for node in decoder_nodes:
                    node.update_period = 0
                for node in code_nodes:
                    node.update_period = 1
                tree.inference_mode = "EM"
                tree.update_type = "parabolic"
                tree.acceleration_start_iter = 3
                tree.cost_computation_iter = 10
                tree.stop_estim_threshold = 1e-3
                tree.reset()
                tree.estimate_param(200)
                print("iter encoding", tree.current_iter)

                # then we update decoder
                for node in decoder_nodes:
                    node.update_period = 1
                    node.inertia = 0.99
                for node in code_nodes:
                    node.update_period = 0
                tree.inference_mode = "VB-MCMC"
                tree.update_type = "regular"
                tree.acceleration_start_iter = 0
                tree.cost_computation_iter = 0
                tree.stop_estim_threshold = 0
                tree.reset()
                tree.estimate_param(iter_per_batch)
        return self._model_from_tree(tree, update_prior=False, snapshot=True)
