from __future__ import annotations

import typing
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import numpy.random as npr
import numpy.typing as npt
import scipy.stats as scs
from tqdm import tqdm

import wonterfact as wtf

param_name = typing.Literal["sign", "coef", "zoom_out"]
rng = npr.default_rng()


def logmeanexp(vector):
    max_val = np.max(vector)

    log_sum_exp = np.log(np.sum(np.exp(vector - max_val)))

    return log_sum_exp + max_val - np.log(vector.size)


class ShrinkLayer(typing.TypedDict):
    type: typing.Literal["shrink"]
    size: int
    shrink_prior: npt.NDArray


class DenseLayer(typing.TypedDict):
    type: typing.Literal["dense"]
    size: int
    sign_prior: npt.NDArray
    covar_prior: npt.NDArray
    shrink_prior: npt.NDArray


class CodeLayer(typing.TypedDict):
    type: typing.Literal["code"]
    size: int
    code_prior: npt.NDArray


Layer = DenseLayer | CodeLayer | ShrinkLayer


@dataclass
class ToyModel:
    code_prior: npt.NDArray
    shrink_prior: npt.NDArray

    def make_tree(
        self,
        data: npt.NDArray,
        update_decoder: bool,
        inference: typing.Literal["VB-MCMC", "EM"] = "VB-MCMC",
    ) -> wtf.Root:
        assert data.shape[1] == 2
        assert data.ndim == 2
        obs_nb = data.shape[0]
        tensor_init = np.ones((obs_nb, 2))
        tensor_init[:, ...] = self.code_prior
        tensor_init /= tensor_init.sum(1, keepdims=True)
        code_leaf_ik = wtf.LeafDirichlet(
            tensor=tensor_init,
            index_id="ik",
            norm_axis=(1,),
            name="code",
            prior_shape=self.code_prior,
            variance_factor=1,
        )

        prior_shape = self.shrink_prior.copy()
        tensor_init = prior_shape / prior_shape.sum(1, keepdims=True)  # taking the mean
        zoom_out_leaf_kz = wtf.LeafDirichlet(
            tensor=tensor_init,
            index_id="kz",
            norm_axis=(1,),
            name="zoom_out_0",
            prior_shape=prior_shape,
            inertia=0.0,
            brake=0,
            variance_factor=1.0,
            update_period=1 if update_decoder else 0,
        )
        if not update_decoder:
            # no need for decoder prior update if decoder is fixed
            zoom_out_leaf_kz.shape_parent.update_period = 0

        mul_nz = wtf.Multiplier(index_id="iz", name="data_hat")
        mul_nz.new_parents(code_leaf_ik, zoom_out_leaf_kz)

        obs_nz = wtf.PosObserver(tensor=data, index_id="iz", norm_axis=(1,), name="data")
        obs_nz.new_parent(mul_nz)

        tree = wtf.Root(
            inference_mode=inference,
            cost_computation_iter=0,
            update_type="regular",
            verbose_iter=0,
            stop_estim_threshold=0,
        )
        tree.new_parent(obs_nz)
        return tree

    def run_vbmcmc(
        self,
        data: npt.NDArray,
        *,
        update_decoder: bool = True,
        n_burnin: int = 1000,
        n_mcmc: int = 10000,
        callback: typing.Callable[[wtf.Root], None] | None = None,
    ):
        tree = self.make_tree(data, update_decoder=update_decoder)
        tree.estimate_param(n_burnin, callback=callback)
        tree.tree_traversal("reset_sufficient_statistic", "bottom-up")
        tree.estimate_param(n_mcmc, callback=callback)
        return tree

    def prelearn(self, dataset: npt.NDArray, learning_rate: float):
        tree = self.run_vbmcmc(dataset)
        if learning_rate >= 0.9:
            tree.estimate_hyperparam(10000, learning_rate=learning_rate)
        else:
            tree.estimate_hyperparam(1000, learning_rate=learning_rate)
        new_model = deepcopy(self)
        new_model.shrink_prior = tree.get_leaf("zoom_out_0").prior_shape
        new_model.code_prior = tree.get_leaf("code").prior_shape[0]
        return new_model

    def fit(self, data: npt.NDArray, code_prior_learning_rate: float = 1.0):
        new_model = deepcopy(self)
        # We estimate the decoder posterior
        # cost = []
        # param1 = []
        # param2 = []

        # def callback(_tree: wtf.Root):
        #     cost.append(_tree.get_cost_func())
        #     leaf = _tree.get_leaf("zoom_out_0")
        #     param1.append(leaf.tensor[0, 0])
        #     param2.append(leaf.tensor[1, 0])
        callback = None

        tree = self.run_vbmcmc(
            data, update_decoder=True, n_burnin=1000, n_mcmc=10000, callback=callback
        )
        # update decoder prior (we are not interested in code posterior yet)
        tree.get_leaf("code").shape_parent.update_period = 0
        tree.estimate_hyperparam(10000)
        new_model.shrink_prior = tree.get_leaf("zoom_out_0").prior_shape

        # with fix decoder, estimate code posterior and set as a new prior
        # tree = new_model.make_tree(data, update_decoder=False, inference="EM")
        # tree.estimate_param(1000)
        tree2 = new_model.run_vbmcmc(
            data, update_decoder=False, n_burnin=100, n_mcmc=1000, callback=None
        )
        tree2.estimate_hyperparam(400, learning_rate=code_prior_learning_rate)
        new_model.code_prior = tree2.get_leaf("code").prior_shape[0]
        # return new_model, tree, param1, param2, cost
        return new_model

    def estimate_log_evidence(self, data: npt.NDArray, n_is: int = 2000):
        assert data.shape == (2,)
        code = []
        cost = []

        def callback(_tree: wtf.Root):
            code.append(_tree.get_leaf("code").tensor[0, 0])
            cost.append(_tree.get_cost_func())

        tree = self.run_vbmcmc(
            np.atleast_2d(data), update_decoder=False, n_burnin=100, n_mcmc=1000, callback=callback
        )
        # code posterior can be very peacky when only one observation: we need many iterations
        tree.estimate_hyperparam(1000)

        # get define code prior and posterior
        code_posterior = tree.get_leaf("code").prior_shape[0]

        # remake tree with prior
        tree = self.make_tree(np.atleast_2d(data), update_decoder=False)
        tree.estimate_param(1)
        var_list = [0.25, 0.5, 0.75, 1, 1.5]
        # var_list = [0.25, 0.5, 1.0]
        q_proba_list = [scs.dirichlet(code_posterior * var) for var in var_list]
        l_arr = np.zeros((len(var_list), n_is))
        for qq, q_proba in tqdm(enumerate(q_proba_list), total=len(q_proba_list), disable=True):
            weights_to_test = q_proba.rvs(size=n_is)
            for ii, weights in enumerate(weights_to_test):
                tree.get_leaf("code").tensor[...] = weights[...]
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
                    np.array([qq_proba.logpdf(weights) for qq_proba in q_proba_list])
                )
        return logmeanexp(l_arr.ravel())

    def estimate_log_evidence2(self, data: npt.NDArray):
        # assert data.shape == (2,)
        # tree = self.make_tree(np.atleast_2d(data), update_decoder=False)
        # tree.estimate_param(100)
        # alpha_post = tree.nodes_by_id["code"].tensor * int(data.sum())
        # alpha_prior = tree.nodes_by_id["code"].prior_shape
        # var_list = [0, 1 / 100, 1 / 50, 1 / 10, 1 / 5, 1]
        # q_proba_list = [scs.dirichlet((alpha_post * var + alpha_prior).ravel()) for var in var_list]
        # N_per_q = 2000
        # R_arr = []
        # for q_proba in q_proba_list:
        #     weights_to_test = q_proba.rvs(size=N_per_q)
        #     for weights in weights_to_test:
        #         tree.nodes_by_id["code"].tensor[...] = weights[...]
        #         tree.tree_traversal(
        #             "_update_tensor",
        #             mode="top-down",
        #             method_input=((), {"update_type": "no_update_for_leaves"}),
        #             iteration_number=1,
        #             type_filter_list=[
        #                 wtf.BudShape,
        #             ],
        #         )
        #         R = np.exp(-tree.get_cost_func()) / (
        #             sum(qq_proba.pdf(weights) for qq_proba in q_proba_list) / len(q_proba_list)
        #         )
        #         R_arr.append(R)
        # return np.mean(R_arr)
        pass


def toy_model(code_prior: npt.NDArray, shrink_prior: npt.NDArray):
    code_layer: CodeLayer = {"type": "code", "size": 1, "code_prior": code_prior}
    shrink_layer: ShrinkLayer = {"type": "shrink", "size": 1, "shrink_prior": shrink_prior}
    return MyModel([code_layer, shrink_layer])


@dataclass
class MyModel:
    layers: list[Layer] = field(default_factory=list)

    @property
    def code_layer(self):
        if not self.layers:
            msg = "no code layer yet"
            raise AttributeError(msg)
        layer = self.layers[0]
        assert layer["type"] == "code"
        return layer

    def add_dense_layer(
        self,
        size: int,
        sign_prior: npt.NDArray | None = None,
        covar_prior: npt.NDArray | None = None,
        shrink_prior: npt.NDArray | None = None,
    ) -> None:
        if not self.layers:
            msg = "The first layer should be a code layer"
        previous_size = self.layers[-1]["size"]
        temp_dict = {}
        for prior_name, prior_arr, expected_size in zip(
            ("sign_prior", "covar_prior", "shrink_prior"),
            (sign_prior, covar_prior, shrink_prior),
            ((size, previous_size, 2), (size, previous_size), (size, 2, 2)),
        ):
            temp_dict[prior_name] = np.ones(expected_size) if prior_arr is None else prior_arr
            if temp_dict[prior_name].shape != expected_size:
                msg = f"wrong shape for `{prior_name}` array"
                raise ValueError(msg)
        self.layers.append(
            {
                "type": "dense",
                "size": size,
                "sign_prior": temp_dict["sign_prior"],
                "covar_prior": temp_dict["covar_prior"],
                "shrink_prior": temp_dict["shrink_prior"],
            }
        )

    def add_code_layer(self, size: int, code_prior: npt.NDArray | None = None) -> None:
        if self.layers:
            msg = "Code layer can only be added once, at the beginning"
            raise ValueError(msg)
        code_prior = np.ones((size, 2)) if code_prior is None else code_prior
        if code_prior.shape != (size, 2):
            msg = "shape of `code_prior` should be (`size`, 2)"
            raise ValueError(msg)
        self.layers.append({"type": "code", "size": size, "code_prior": code_prior})

    def _get_layer(self, layer_nb: int) -> Layer:
        if layer_nb >= len(self.layers) or layer_nb < 0:
            msg = f"Layer number {layer_nb} does not exist"
            raise ValueError(msg)
        return self.layers[layer_nb]

    def make_layer(self, layer_nb: int, update_decoder: bool):
        layer = self._get_layer(layer_nb)
        if layer["type"] == "dense":
            return self._make_dense_layer(layer, layer_nb, update_decoder=update_decoder)
        if layer["type"] == "shrink":
            return self._make_shrink_layer(layer, layer_nb, update_decoder=update_decoder)
        msg = "unknwon layer type"
        raise ValueError(msg)

    def get_layer_from_tree(self, tree: wtf.Root, layer_nb: int) -> Layer:
        layer = self._get_layer(layer_nb)
        if layer["type"] == "dense":
            return self._get_dense_layer(tree, layer_nb)
        if layer["type"] == "shrink":
            return self._get_shrink_layer(tree, layer_nb)
        if layer["type"] == "code":
            return self._get_code_layer(tree)
        raise NotImplementedError

    @staticmethod
    def prior_to_leaf_name(prior_name: str):
        return prior_name.split("_prior")[0]

    @staticmethod
    def _make_dense_layer(layer: DenseLayer, layer_nb: int, update_decoder: bool):
        update_period = 1 if update_decoder else 0
        prior = layer["sign_prior"]
        sign = wtf.LeafDirichlet(
            index_id="fgs",
            norm_axis=(2,),
            name=f"sign_{layer_nb}",
            tensor=prior / prior.sum(2, keepdims=True),
            prior_shape=prior,
            update_period=update_period,
        )
        prior = layer["covar_prior"]
        covar = wtf.LeafDirichlet(
            index_id="fg",
            norm_axis=(1,),
            name=f"covar_{layer_nb}",
            tensor=prior / prior.sum(1, keepdims=True),
            prior_shape=prior,
            update_period=update_period,
        )
        signed_covar = wtf.Multiplier(
            index_id="fgs", name=f"signed_covar_{layer_nb}", update_period=update_period
        )
        signed_covar.new_parents(sign, covar)
        plug_here = wtf.Multiplier(index_id="nfzs", name=f"plug_here_{layer_nb}")
        plug_here.new_parent(signed_covar)
        sign_flipper = wtf.LeafDirichlet(
            index_id="zsw",
            tensor=np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            update_period=0,
            name=f"flipper_{layer_nb}",
        )
        after_rotation = wtf.Multiplier(index_id="nfw", name=f"rotated_code_{layer_nb}")
        after_rotation.new_parents(plug_here, sign_flipper)
        prior = layer["shrink_prior"]
        zoom_out = wtf.LeafDirichlet(
            index_id="fwz",
            norm_axis=(2,),
            name=f"shrink_{layer_nb}",
            tensor=prior / prior.sum(2, keepdims=True),
            prior_shape=prior,
            update_period=update_period,
        )
        after_zoom_out = wtf.Multiplier(index_id="nfz", name=f"zoom_out_code_{layer_nb}")
        after_zoom_out.new_parents(zoom_out, after_rotation)
        output = wtf.Smoothstep(index_id="ngz", name=f"output_{layer_nb}")
        after_zoom_out.new_child(output, index_id_for_child="ngz")
        return plug_here, output

    @staticmethod
    def _make_shrink_layer(layer: ShrinkLayer, layer_nb: int, update_decoder: bool):
        prior = layer["shrink_prior"]
        shrinking = wtf.LeafDirichlet(
            index_id="gzv",
            norm_axis=(2,),
            name=f"shrink_{layer_nb}",
            tensor=prior / prior.sum(2, keepdims=True),
            prior_shape=prior,
            update_period=int(update_decoder),
        )
        after_zoom_out = wtf.Multiplier(index_id="ngv", name=f"zoom_out_code_{layer_nb}")
        after_zoom_out.new_parent(shrinking)
        output = wtf.Proxy(index_id="ngz")
        after_zoom_out.new_child(output, index_id_for_child="ngz")
        return after_zoom_out, output

    @staticmethod
    def _get_dense_layer(tree: wtf.Root, layer_nb: int) -> DenseLayer:
        try:
            sign_prior = tree.get_leaf(f"sign_{layer_nb}").shape_parent.tensor_as_numpy
            covar_prior = tree.get_leaf(f"covar_{layer_nb}").shape_parent.tensor_as_numpy
            shrink_prior = tree.get_leaf(f"shrink_{layer_nb}").shape_parent.tensor_as_numpy
        except ValueError as err:
            msg = "Requested dense layer does not exist"
            raise ValueError(msg) from err
        layer: DenseLayer = {
            "type": "dense",
            "size": covar_prior.shape[0],
            "covar_prior": covar_prior,
            "sign_prior": sign_prior,
            "shrink_prior": shrink_prior,
        }
        return layer

    @staticmethod
    def _get_shrink_layer(tree: wtf.Root, layer_nb: int) -> ShrinkLayer:
        try:
            shrink_prior = tree.get_leaf(f"shrink_{layer_nb}").shape_parent.tensor_as_numpy
        except ValueError as err:
            msg = "Requested shrink layer does not exist"
            raise ValueError(msg) from err
        layer: ShrinkLayer = {
            "type": "shrink",
            "size": shrink_prior.shape[0],
            "shrink_prior": shrink_prior,
        }
        return layer

    @staticmethod
    def _get_code_layer(tree: wtf.Root) -> CodeLayer:
        try:
            code_prior = tree.get_leaf("code").shape_parent.tensor_as_numpy
        except ValueError as err:
            msg = "Code layer does not exist"
            raise ValueError(msg) from err
        layer: CodeLayer = {"type": "code", "size": code_prior.shape[0], "code_prior": code_prior}
        return layer

    def make_code_layer(self, batch_size: int = 1):
        if not self.layers:
            msg = "please specify a code layer"
            raise ValueError(msg)
        layer = self.layers[0]
        if layer["type"] != "code":
            msg = "First layer should be a code layer"
            raise ValueError(msg)
        tensor = np.ones((batch_size, layer["size"], 2))
        tensor[:, ...] = layer["code_prior"]
        tensor = tensor / tensor.sum(-1, keepdims=True)
        return wtf.LeafDirichlet(
            index_id="ngz",
            norm_axis=(2,),
            tensor=tensor,
            prior_shape=layer["code_prior"],
            name="code",
        )

    def make_tree(self, data: npt.NDArray, update_decoder: bool):
        if data.ndim != 3 or data.shape[-1] != 2:
            msg = "Shape of data array should be (batch_size, data_dimension, 2)"
            raise ValueError(msg)
        if data.shape[1] != self.layers[-1]["size"]:
            msg = "size of last layer should correspond to data dimension"
            raise ValueError(msg)
        batch_size = data.shape[0]
        to_plug = self.make_code_layer(batch_size)
        for layer_nb in range(1, len(self.layers)):
            plug_here, output = self.make_layer(layer_nb, update_decoder=update_decoder)
            to_plug.new_child(plug_here)
            to_plug = output

        obs_ngz = wtf.PosObserver(tensor=data, index_id="ngz", norm_axis=(2,), name="data")
        to_plug.new_child(obs_ngz)

        root = wtf.Root(
            inference_mode="VB-MCMC",
            cost_computation_iter=0,
            update_type="regular",
            verbose_iter=0,
            stop_estim_threshold=0,
        )
        obs_ngz.new_child(root)
        return root

    def __repr__(self) -> str:
        str_repr = type(self).__name__ + "(\n"
        for layer in self.layers:
            str_repr += f"    [{layer['type']} layer, output size {layer['size']}]\n"
        str_repr += ")"
        return str_repr

    def run_vbmcmc(
        self,
        data: npt.NDArray,
        *,
        update_decoder: bool = True,
        n_burnin: int = 1000,
        n_mcmc: int = 10000,
        callback: typing.Callable[[wtf.Root], None] | None = None,
    ):
        tree = self.make_tree(data, update_decoder=update_decoder)
        tree.estimate_param(n_burnin, callback=callback)
        tree.tree_traversal("reset_sufficient_statistic", "bottom-up")
        tree.estimate_param(n_mcmc, callback=callback)
        return tree

    def _model_from_tree(self, tree: wtf.Root):
        new_layers: list[Layer] = [
            self.get_layer_from_tree(tree, layer_nb) for layer_nb in range(len(self.layers))
        ]
        return type(self)(new_layers)

    def prelearn(self, dataset: npt.NDArray, learning_rate: float):
        tree = self.run_vbmcmc(dataset)
        if learning_rate >= 0.9:
            tree.estimate_hyperparam(10000, learning_rate=learning_rate)
        else:
            tree.estimate_hyperparam(1000, learning_rate=learning_rate)
        return self._model_from_tree(tree)

    def fit(self, dataset: npt.NDArray, code_prior_learning_rate: float = 1.0):
        # We estimate the decoder posterior
        # cost = []
        # param1 = []
        # param2 = []

        # def callback(_tree: wtf.Root):
        #     cost.append(_tree.get_cost_func())
        #     leaf = _tree.get_leaf("zoom_out_0")
        #     param1.append(leaf.tensor[0, 0])
        #     param2.append(leaf.tensor[1, 0])
        callback = None

        tree = self.run_vbmcmc(
            dataset, update_decoder=True, n_burnin=1000, n_mcmc=10000, callback=callback
        )
        # update decoder prior (we are not interested in code posterior yet)
        tree.get_leaf("code").shape_parent.update_period = 0
        tree.estimate_hyperparam(10000)

        model_with_fix_code = self._model_from_tree(tree)

        # with fix decoder, estimate code posterior and set as a new prior
        # tree = new_model.make_tree(data, update_decoder=False, inference="EM")
        # tree.estimate_param(1000)
        tree2 = model_with_fix_code.run_vbmcmc(
            dataset, update_decoder=False, n_burnin=100, n_mcmc=1000, callback=None
        )
        tree2.estimate_hyperparam(400, learning_rate=code_prior_learning_rate)
        return self._model_from_tree(tree2)

    def estimate_log_evidence(self, datapoint: npt.NDArray, n_is: int = 2000):
        # code = []
        # cost = []

        # def callback(_tree: wtf.Root):
        #     code.append(_tree.get_leaf("code").tensor[0, 0])
        #     cost.append(_tree.get_cost_func())
        callback = None

        tree = self.run_vbmcmc(
            datapoint[None, ...], update_decoder=False, n_burnin=100, n_mcmc=1000, callback=callback
        )
        # code posterior can be very peacky when only one observation: we need many iterations
        tree.estimate_hyperparam(1000)

        # get define code prior and posterior
        code_posterior = tree.get_leaf("code").shape_parent.tensor_as_numpy

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
            code_to_test = np.empty((n_is, self.code_layer["size"], 2))
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
