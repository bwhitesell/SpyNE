import numpy as np

from spyne.operations.base import UniTensorOperation, DualTensorOperation
from spyne.data_structures import Tensor
from .utils import basis_vectors


class BackwardsPass:
    var_internal_shapes = {}
    partials = {}
    _partials = {}

    def __init__(self, var):
        self.var = var
        self.ind_vars = []
        self.nodes = [self.var]
        self.jac_external_shape = self.var.shape if self.var.shape != (1,) else ()
        self.vjps = self._build_vjps()

    def jacobians(self, update=False, alpha=.01):
        jac = {}
        for node_uid, node in self.vjps.items():
            g = [node[1](b) for b in basis_vectors(self.var)]
            jac[node_uid] = np.reshape(
                g,
                self.jac_external_shape + node[0].shape
            )
            if update:
                node[0].value += jac[node_uid] * alpha
        return jac

    def _build_vjps(self):
        self._partials = {
            self.var.node_uid: lambda g: g
        }
        vjps = {}
        for node in self.nodes:

            node_vjp = self._partials[node.node_uid]
            self.var_internal_shapes[node.node_uid] = node.shape

            if issubclass(node.__class__, UniTensorOperation):
                g_a = node.vector_jacobian_product(node_vjp)
                self._add_node(node.a, g_a)

            elif issubclass(node.__class__, DualTensorOperation):
                g_a, g_b = node.vector_jacobian_product(node_vjp)

                self._add_node(node.a, g_a)
                self._add_node(node.b, g_b)

            elif issubclass(node.__class__, Tensor):
                vjps[node.node_uid] = (node, self._partials[node.node_uid])

        return vjps

    def _add_node(self, node, vjp):
        self.nodes.append(node)
        if node.node_uid not in self._partials:
            self._partials[node.node_uid] = vjp
        else:
            f_prior = self._partials[node.node_uid]
            self._partials[node.node_uid] = lambda x: vjp(x) + f_prior(x)
