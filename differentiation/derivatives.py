import numpy as np

from operations.base import UniTensorOperation, DualTensorOperation
from variables.variables import Tensor
from .utils import basis_vectors


class Gradient:
    var_internal_shapes = {}
    partials = {}
    _partials = {}

    def __init__(self, var):
        self.var = var
        self.ind_vars = []
        self.nodes = [self.var]
        self.jac_external_shape = self.var.shape

    @property
    def jacobians(self):
        jac = {}
        vjps = self.build_vjps()

        for node, vjp in vjps.items():
            g = [vjp(b) for b in basis_vectors(self.var)]
            jac[node] = np.reshape(
                g,
                self.jac_external_shape + self.var_internal_shapes[node]
            )

        return jac

    def build_vjps(self):
        _partials = {}
        vjps = {}
        for node in self.nodes:
            self.var_internal_shapes[node.node_uid] = node.shape

            if issubclass(node.__class__, UniTensorOperation):
                self.nodes.append(node.a)
                g_a = node.vector_jacobian_product()

                if node.a not in _partials:
                    _partials[node.a.node_uid] = g_a
                else:
                    _partials[node.a.node_uid] = lambda g: g_a(g) + _partials[node.a.node_uid](g)

            elif issubclass(node.__class__, DualTensorOperation):
                self.nodes.append(node.a)
                self.nodes.append(node.b)
                g_a, g_b = node.vector_jacobian_product()
                if node.a not in _partials:
                    _partials[node.a.node_uid] = g_a
                else:
                    _partials[node.a.node_uid] = lambda g: g_a(g) + _partials[node.a.node_uid](g)
                if node.b not in _partials:
                    _partials[node.b.node_uid] = g_b
                else:
                    _partials[node.b.node_uid] = lambda g: g_b(g) + _partials[node.b.node_uid](g)

            elif issubclass(node.__class__, Tensor):
                vjps[node.node_uid] = _partials[node.node_uid]

        return vjps
