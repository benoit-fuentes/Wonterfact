# ----------------------------------------------------------------------------
# Copyright 2020 Smart Impulse SAS, Benoit Fuentes <bf@benoit-fuentes.fr>
#
# This file is part of Wonterfact.
#
# Wonterfact is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Wonterfact is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wonterfact. If not, see <https://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

"""Module for all leave classes"""

# Python System imports
from functools import cached_property

# Third-party imports
import numpy as np

# Relative imports
from . import utils, core_nodes, buds
from .glob_var_manager import glob


class LeafDirichlet(core_nodes._DynNodeData, core_nodes._ChildNode):
    """
    Class for the Dirichlet leaves of a graphical model, i.e. normalized tensors to estimate in a
    factorization model.
    """

    def __init__(
        self,
        norm_axis: tuple[int] = (-1,),
        brake: None | float = None,
        inertia: float = 0,
        variance_factor: float = 1.0,
        init_type="custom",
        prior_shape=None,
        constraint_coeffs=None,
        constraint_type="inequality",
        constraint_max_iter=10,
        prior_accelerator=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        norm_axis: sequence of int, optional, default ()
            Normalization axis for inner tensor, such that
            `self.tensor.sum(norm_axis) == 1` is all True. Must be the last axes
            of tensor.
        init_type: 'custom', 'prior', or 'random', default 'custom'
            If 'prior', initialization is defined by the prior distribution
            (mode of prior in EM mode and 'exp(mean of sufficient statistic)' in
            VBEM mode); if 'custom', inner tensor is initialized by the user via
            the 'tensor' attribute; if 'random', inner tensor is randomly
            initialized.
        prior_shape: array_like or float or None, optional, default None
            Shape hyperparameter of the prior distribution. If not None, automatically
            creates a BudShape node and links it to the returned leaf.
        constraint_coeffs: array_like or None, optional, default None
            If None, no constraint is applied. Otherwise, coefficients for
            linear equality (resp. inequality) constraints. Inner tensor will
            always comply to the constraint `(self.tensor *
            self.constraint_coeffs).sum(axis_to_sum) == 0` (resp `>=0`)
            where `axis_to_sum` are the `ndim` last axes of `self.tensor` and
            `ndim=self.constraint_coeffs.ndim`.
        constraint_type: 'equality' or 'inequality', optional, default 'inequality'
            Defines the type of linear constraint to apply if
            `constraint_coeffs` is provided.
        constraint_max_iter: int, optional, default 10
            Maximum number of iterations for the inner algorithm used to comply
            to the linear constraints. The greater it is, the more precise it is
            but also the slower.
        prior_accelerator: float or None, optional, default None
            Allows to give more importance to the priors without having to
            change their value. Useful for accelerating sparsity prior in the
            'VBEM' mode, when `prior_shape < 1`
        """
        self._norm_axis = norm_axis
        self.brake = brake
        self.inertia = inertia
        self.variance_factor = variance_factor
        self.init_type = init_type
        self.constraint_coeffs = constraint_coeffs
        self.constraint_type = constraint_type
        self.constraint_max_iter = constraint_max_iter
        self.prior_accelerator = prior_accelerator
        if self.constraint_coeffs is not None:
            self.constraint_coeffs = glob.xp.array(
                self.constraint_coeffs, dtype=glob.float
            )
        self._set_inference_mode()
        super().__init__(**kwargs)
        if prior_shape is not None:
            self._create_bud_parent(prior_shape)
        self.last_tensor_update = None
        self.n_sufficient_statistic = 0
        self.tensor_sufficient_statistic = glob.xp.zeros_like(self.tensor)

    @property
    def min_val(self):
        if self._inference_mode == "VBEM":
            return 0.02
        return 1e-20

    @property
    def norm_axis(self):
        return self._norm_axis

    @property
    def level(self):
        """
        Returns 1 , which is the default level for of a leaf.

        Returns
        ------
        int
        """
        return 1

    def _create_bud_parent(self, prior_val):
        """
        Automatically creates a BudShape
        """
        prior_val = glob.xp.array(prior_val, dtype=glob.float)
        ndim = prior_val.ndim
        idx = []
        shape = []
        for num_dim in range(ndim):
            if (
                prior_val.shape[-num_dim - 1] != 1
                or prior_val.shape[-num_dim - 1] == self.tensor.shape[-num_dim - 1]
            ):
                idx.append(self.index_id[-num_dim - 1])
                shape.append(self.tensor.shape[-num_dim - 1])
        idx = tuple(idx[::-1])
        shape = tuple(shape[::-1])
        if isinstance(self.index_id, str):
            idx = "".join(idx)
        update_period = 1 if self.update_period else 0
        bud = buds.BudShape(
            name="{}_{}".format(self.name, "shape"),
            index_id=idx,
            tensor=prior_val.reshape(shape),
            update_period=update_period,
        )
        bud.new_child(self)

    @property
    def prior_shape(self):
        if self.shape_parent is not None:
            return self._get_prior_arr(self.shape_parent)
        else:
            return glob.xp.array(1.0)

    def _get_prior_arr(self, prior_parent):
        transpose, sl = utils.get_transpose_and_slice(
            prior_parent.get_index_id_for_children(self), self.index_id
        )
        return prior_parent.get_tensor_for_children(self).transpose(transpose)[sl]

    def _clip_tensor_min_value(self):
        if self._inference_mode == "VBEM":
            utils.clip_inplace(
                self.posterior_shape, a_min=self.min_val, backend=glob.backend
            )
        else:
            utils.clip_inplace(self.tensor, a_min=self.min_val, backend=glob.backend)

    def _initialization(self):
        if self.update_period != 0:
            self.tensor_update = glob.xp.empty_like(self.tensor)
            if self._inference_mode == "VBEM":
                self.posterior_shape = self.tensor.copy() * 1e5
        if self.update_period == 0 or self._inference_mode in ("EM", "VB-MCMC"):
            if not glob.xp.allclose(self.tensor.sum(axis=self.norm_axis), 1):
                raise ValueError("Please provide a well normalized tensor")

    def _update_tensor(self, update_type="regular", update_param=None):
        # update rules are the same in VBEM and EM mode
        if update_type == "no_update_for_leaves":
            pass
        elif update_type == "just_normalize":
            self._normalize_tensor()
        elif update_type == "parabolic":
            self._parabolic_update(parabolic_param=update_param)
        elif update_type == "regular":
            self._regular_update_tensor()
        else:
            raise ValueError("Unknown `update_type`")

    def _normalize_tensor(self, **kwargs):
        if self.constraint_coeffs is not None and self._inference_mode == "VBEM":
            raise NotImplementedError

        if self._inference_mode == "VB-MCMC" and self.variance_factor != 0:
            self.tensor[...] = glob.xp.random.gamma(self.tensor - 0.5, 1)
            # self.tensor[...] = glob.xp.random.gamma(self.tensor, 1)
            self._clip_tensor_min_value()

        if self._inference_mode in ("EM", "VB-MCMC") or self.update_period == 0:
            norm_tensor = self.tensor.sum(axis=self.norm_axis, keepdims=True)
            if self.constraint_coeffs is not None:
                sigma = utils._find_equality_root(
                    self.tensor,
                    norm_tensor,
                    self.constraint_coeffs,
                    self.constraint_max_iter,
                    type=self.constraint_type,
                    atol=1e-10,
                )
                self.tensor /= norm_tensor - sigma * self.constraint_coeffs
            else:
                self.tensor /= norm_tensor
        elif self._inference_mode in "VBEM":
            norm_tensor = self.posterior_shape.sum(axis=self.norm_axis, keepdims=True)
            self.tensor[...] = utils.exp_digamma(
                self.posterior_shape
            ) / utils.exp_digamma(norm_tensor)

    def _regular_update_tensor(self):
        if self.prior_accelerator is not None:
            self.tensor[...] = self.tensor / self.prior_accelerator

        if self.last_tensor_update is not None and self.inertia:
            tensor_update = self.inertia * self.last_tensor_update + self.tensor_update
        else:
            tensor_update = self.tensor_update
        if self.brake:
            tensor_update = tensor_update + self.brake

        if self.variance_factor not in [0.0, 1.0]:
            tensor_update /= self.variance_factor

        if self._inference_mode == "EM":
            self.tensor *= tensor_update
            self.tensor += self.prior_shape - 1
            self._clip_tensor_min_value()

        if self._inference_mode == "VB-MCMC":
            self.tensor *= tensor_update
            self.tensor += self.prior_shape

        if self._inference_mode == "VBEM":
            self.posterior_shape[...] = self.tensor * tensor_update
            self.posterior_shape += self.prior_shape

        self._normalize_tensor()
        if self.inertia:
            self.last_tensor_update = tensor_update
        if self._inference_mode == "VB-MCMC":
            self.update_sufficient_statistic()

    def update_sufficient_statistic(self):
        self.tensor_sufficient_statistic += glob.xp.log(self.tensor)
        self.n_sufficient_statistic += 1

    def reset_sufficient_statistic(self):
        self.tensor_sufficient_statistic[...] = 0
        self.n_sufficient_statistic = 0

    def _set_bezier_point(self, param):
        if self._inference_mode == "EM":
            tensor = self.tensor
        elif self._inference_mode == "VBEM":
            tensor = self.posterior_shape
        else:
            msg = f"Not possible for this inference mode {self._inference_mode}"
            raise NotImplementedError(msg)
        if glob.processor == glob.GPU:
            utils.xp_utils.get_cupy_utils(glob.backend)._set_bezier_point(
                self._past_tensor[0],
                self._past_tensor[1],
                self._past_tensor[2],
                param,
                tensor,
            )
        else:
            tensor[...] = (
                (1 - param) ** 2 * self._past_tensor[0]
                + 2 * (1 - param) * param * self._past_tensor[1]
                + (param**2) * self._past_tensor[2]
            )

    def _parabolic_update(self, parabolic_param):
        self._set_bezier_point(parabolic_param)
        if self._inference_mode == "EM":
            if (self.tensor <= self.min_val).any():
                self._clip_tensor_min_value()
                self._normalize_tensor()

        # if VBEM, one need to recompute self.tensor
        if self._inference_mode == "VBEM":
            self._clip_tensor_min_value()
            self._normalize_tensor()

    def _update_past_tensors(self):
        if not hasattr(self, "_past_tensor"):
            self._past_tensor = [glob.xp.empty_like(self.tensor) for __ in range(3)]
        if self._inference_mode == "EM":
            self._past_tensor[0][...] = self.tensor.copy()
        elif self._inference_mode == "VBEM":
            self._past_tensor[0][...] = self.posterior_shape.copy()
        else:
            raise ValueError("unknown inference mode")
        self._past_tensor = self._past_tensor[1:] + self._past_tensor[:1]

    @property
    def _might_need_clipping(self):
        return (self.prior_shape <= 1).any()

    @property
    def _prior_alpha_all_one(self):
        return (self.shape_parent is None) or (self.prior_shape == 1).all()

    @cached_property
    def shape_parent(self):
        return next(
            (
                parent
                for parent in self.list_of_parents
                if isinstance(parent, buds.BudShape)
            ),
            None,
        )

    def get_posterior_shape(self, force_numpy=False):
        """
        Return the posterior shape array. If force_numpy is true, the tensor is casted to
        numpy ndarray if needed (if cupy backend is used)
        """
        return self.cast_array(self.posterior_shape, force_numpy=force_numpy)

    def get_posterior_rate(self, force_numpy=False):
        """
        Return the posterior rate array. If force_numpy is true, the tensor is casted to
        numpy ndarray if needed (if cupy backend is used)
        """
        return self.cast_array(self.posterior_rate, force_numpy=force_numpy)

    @property
    def _cst_prior_value(self):
        if self._inference_mode in ("EM", "VB-MCMC"):
            prior_shape = glob.xp.zeros_like(self.tensor) + self.prior_shape
            cst_prior = (
                glob.sps.gammaln((prior_shape).sum(self.norm_axis)).sum()
                - glob.sps.gammaln(prior_shape).sum()
            )
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            cst_prior = (
                glob.sps.gammaln(prior_shape.sum(self.norm_axis, keepdims=True)).sum()
                - glob.sps.gammaln(prior_shape).sum()
            )
        if self.prior_accelerator is not None:
            cst_prior *= self.prior_accelerator
        return cst_prior.item()

    def _prior_value(self):
        if self.update_period == 0 or self.norm_axis == ():
            return 0
        if self._inference_mode in ("EM", "VB-MCMC"):
            if self._prior_alpha_all_one:
                prior_val = np.array(0)
            else:
                prior_val = utils.xlogy(self.prior_shape - 1, self.tensor).sum()
        elif self._inference_mode == "VBEM":
            prior_shape = (
                glob.xp.zeros(self.tensor.shape, dtype=glob.float) + self.prior_shape
            )
            prior_val = (
                glob.sps.gammaln(self.posterior_shape).sum()
                - (
                    glob.sps.gammaln(
                        self.posterior_shape.sum(self.norm_axis, keepdims=True)
                    )
                ).sum()
            )
            prior_val -= (
                (self.posterior_shape - prior_shape)
                * (
                    glob.xp.log(utils.exp_digamma(self.posterior_shape))
                    - glob.xp.log(
                        utils.exp_digamma(
                            self.posterior_shape.sum(self.norm_axis, keepdims=True)
                        )
                    )
                )
            ).sum()
        if self.prior_accelerator is not None:
            prior_val *= self.prior_accelerator
        return prior_val.item() + self._cst_prior_value

    def _bump(self):
        if self._inference_mode in ("EM", "VB-MCMC"):
            posterior_shape = self.tensor * self.tensor_update + self.prior_shape
            # we should instead use the VB-MCMC option
        elif self._inference_mode == "VBEM":
            posterior_shape = self.posterior_shape
        self.tensor[...] = glob.xp.random.gamma(
            posterior_shape, glob.xp.ones_like(self.tensor)
        )
        self.tensor /= self.tensor.sum(axis=self.norm_axis, keepdims=True)

    def get_l2_norm(self, **kwargs):
        """
        Returns the l2 norm of each distribution contained in the tensor.
        """
        tensor = self.tensor.reshape(self.tensor.shape[: self.norm_axis[0]] + (-1,))
        return glob.xp.linalg.norm(tensor, ord=2, axis=-1, **kwargs)

    @property
    def tensor_has_energy(self):
        return False

    def _give_update_alpha(self, parent, log_tensor, out=None):
        parent_idx_id = parent.get_index_id_for_children(self)
        update_tensor = utils.einsum(log_tensor, self.index_id, parent_idx_id, out=out)
        if out is None:
            return update_tensor

    def _give_update(self, parent, out):
        if not isinstance(parent, buds.BudShape):
            raise ValueError(
                "Class of parent argument must be wonterfact.bubs.BudShape"
            )
        ## returns quantity e_d (cf technical report)
        if self._inference_mode in ("EM", "VBEM") or self.n_sufficient_statistic == 0:
            log_tensor = glob.xp.log(self.tensor)
        else:
            log_tensor = self.tensor_sufficient_statistic / self.n_sufficient_statistic
        return self._give_update_alpha(parent, log_tensor, out=out)

    def _give_number_of_users(self, parent, out=None):
        """
        Gives the number of parameters that share a same hyperparameter for each
        hyperparameter (corresponds to $|\\phi^{-1}(d)|$ in tech report)
        """
        parent_idx_id = parent.get_index_id_for_children(self)
        number_or_users = utils.einsum(
            glob.xp.ones_like(self.tensor), self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return number_or_users

    def _give_update_bis(self, parent, out=None):
        if not isinstance(parent, buds.BudShape):
            raise ValueError("'parent' must be an instance of wonterfact.bubs.BudShape")
        parent_idx_id = parent.get_index_id_for_children(self)
        prior_tensor = glob.xp.zeros_like(self.tensor) + self.prior_shape
        prior_tensor = glob.xp.zeros_like(self.tensor) + glob.sps.digamma(
            prior_tensor.sum(self.norm_axis, keepdims=True)
        )

        update_tensor = utils.einsum(
            prior_tensor, self.index_id, parent_idx_id, out=out
        )
        if out is None:
            return update_tensor

    def _give_update_first_iteration(self, parent, out=None):
        if not isinstance(parent, buds.BudShape):
            raise ValueError("'parent' must be an instance of wonterfact.bubs.BudShape")
        tensor = np.zeros_like(self.tensor) + self.prior_shape
        log_tensor = glob.xp.log(
            utils.exp_digamma(tensor)
            / utils.exp_digamma(tensor.sum(self.norm_axis, keepdims=True))
        )
        return self._give_update_alpha(parent, log_tensor, out=out)

    def compute_alpha_estim(self, n_iter=10):  # very slow: need research to be faster
        alpha_estim = np.ones_like(self.tensor)
        for __ in range(n_iter):
            alpha_sum = alpha_estim.sum(self.norm_axis, keepdims=True)
            utils.inverse_digamma(
                glob.sps.digamma(alpha_sum) + glob.xp.log(self.tensor), out=alpha_estim
            )
        return alpha_estim
