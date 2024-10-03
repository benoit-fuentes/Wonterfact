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

"""Module for all observer classes"""

from functools import cached_property

from numpy.random import default_rng
import scipy.stats as scs


from . import _core_nodes, utils
from .glob_var_manager import glob


class _Observer(_core_nodes.ChildNode):
    def norm_axis(self):
        parent = self.first_parent
        assert isinstance(parent, _core_nodes.DynNodeData)
        return parent.get_index_id_for_children(self)


class PosObserver(_core_nodes.NodeData, _Observer):  # TODO: optimize mask_data
    """
    Class for nonnegative observations.
    """

    max_parents = 1

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        mask_data: array_like of booleans or None, default None
            Boolean mask to apply to observed 2 to specify which
            coefficients are masked and which are not. If masked, a coefficient
            plays no role in the optimization process.
        drawings_max: float or None, optional, default None
            During optimization algorithm, inner tensor is normalized with a
            coefficient which can increase along iterations up to a limit.
            value. If not None, this limit value is computed as `drawings_max /
            abs(self.tensor).sum()`. If None, the limit value is `1`.
        drawings_update_iter: int, optional, default 1
            The normalization coefficient (see `drawings_max` section) is
            updated every `drawings_update_iter` iterations.
        drawings_step: float or None, optional, default None
            Normalization coefficient is initialized as `drawings_step /
            abs(self.tensor).sum()` and when it has to be updated (see
            `drawings_update_iter`), the same amount is added to the current
            normalization coefficient until it reaches its limit (see
            `drawings_max` section). If None, `drawings_step` is set to
            `drawings_max` value so that the normalization coefficient remains
            fixed during the algorithm.

        Notes
        -----
        The dynamic normalization feature (see `drawings_max`,
        `drawings_update_iter` and `drawings_step` sections) aims at giving more
        weight to the priors in the early stage of the algorithm. If you do not
        want to use this feature, just leave default values for those arguments.
        """
        self.norm_axis = kwargs.pop("norm_axis", ())
        self.mask_data = kwargs.pop("mask_data", None)
        self.drawings_max = kwargs.pop("drawings_max", None)
        self.drawings_step = kwargs.pop("drawings_step", None)
        self.drawings_update_iter = kwargs.pop("drawings_update_iter", 1)
        super().__init__(**kwargs)

        self.drawings_max = self.drawings_max or self.sum_tensor
        self.drawings_step = self.drawings_step or self.sum_tensor
        self.drawings = self.drawings_step
        self.drawings_update_counter = 0

    def _initialization(self):
        pass

    @property
    def number_of_drawings(self):
        return (self.get_current_mult_factor() * self.tensor).sum(self.norm_axis)

    @cached_property
    def is_tensor_null(self):
        return (self.tensor == 0).astype(self.tensor.dtype)

    @cached_property
    def is_tensor_pos(self):
        return self.tensor > 0

    @cached_property
    def sum_tensor(self):
        return glob.xp.abs(self.tensor).sum()

    def there_is_a_mult_factor(self):
        return self.drawings != self.sum_tensor

    def get_current_mult_factor(self):
        return self.drawings / self.sum_tensor if self.sum_tensor else 1

    def update_drawings(self):
        self.drawings_update_counter += 1
        if self.drawings_update_counter % self.drawings_update_iter == 0:
            self.drawings = min(self.drawings_max, self.drawings + self.drawings_step)

    def apply_mask_to_tensor_update(self, tensor_update):
        if self.mask_data is not None:
            tensor_update[self.mask_data] = 1

    def _give_update(self, parent, out=None):
        parent_tensor = parent.get_tensor_for_children(self)

        tensor_update = glob.xp.empty_like(self.tensor) if out is None else out

        denominator = parent_tensor + self.is_tensor_null
        if self._inference_mode == "VBEM":
            # IN VBEM mode, underflow might happen, leading to null parent_tensor
            denominator += parent_tensor == 0

        tensor_update[...] = self.get_current_mult_factor() * self.tensor / denominator

        self.update_drawings()
        self.apply_mask_to_tensor_update(tensor_update)

        return tensor_update

    def get_current_reconstruction(self, parent, force_numpy=False):
        tensor_to_give = parent.get_tensor_for_children(self) / self.get_current_mult_factor()
        if force_numpy and utils.infer_backend(tensor_to_give) == glob.CUPY:
            return glob.xp.asnumpy(tensor_to_give)
        return tensor_to_give

    def get_kl_divergence(self):
        """
        Returns the kullback-Leibler divergence between observer tensor and the
        current reconstruction.
        """
        kl_div = 0
        for parent in self.list_of_parents:
            reconstruction = self.get_current_reconstruction(parent)
            my_tensor = self.tensor
            kl_div -= utils.xlogy(my_tensor, reconstruction).sum()
            kl_div += reconstruction.sum()
            kl_div -= my_tensor.sum() - utils.xlogy(my_tensor, my_tensor).sum()
        return kl_div.item()

    def _get_data_fitting(self):
        """
        Returns minus log-likelihood of Multinomial distribution
        """
        lh = glob.xp.zeros_like(self.tensor)
        for parent in self.list_of_parents:
            my_tensor = self.get_current_mult_factor() * self.tensor
            parent_tensor = parent.get_tensor_for_children(self)
            lh += utils.xlogy(my_tensor, parent_tensor)
            lh -= glob.sps.gammaln(my_tensor + 1)
        if self.mask_data is not None:
            raise NotImplementedError
        return -lh.sum().item() - glob.sps.gammaln(self.number_of_drawings + 1).sum()

    def simulate_data(self, n_drawings):
        parent = self.first_parent
        assert isinstance(parent, _core_nodes.DynNodeData)
        parent_tensor = parent.get_tensor_for_children(self)
        shape_for_multinomial = (
            *(size for ii, size in enumerate(self.tensor.shape) if ii not in self.norm_axis),
            -1,
        )
        rng = default_rng()
        self.tensor[...] = rng.multinomial(
            n_drawings, parent_tensor.reshape(shape_for_multinomial)
        ).reshape(self.tensor.shape)


class BlindObs(_Observer, _core_nodes.ParentNode):
    def _give_update(self, parent, out=None):
        if out is None:
            return glob.xp.ones_like(parent.get_tensor_for_children(self))
        out[...] = 1.0

        return out

    def _get_data_fitting(self):
        if self._inference_mode in ("EM", "VB-MCMC"):
            return 0
        if self._inference_mode == "VBEM":
            total_energy = 0
            for parent in self.list_of_parents:
                total_energy += parent.get_tensor_for_children(self).sum()
            return -total_energy.item()
        msg = "unknown inference mode"
        raise AttributeError(msg)
