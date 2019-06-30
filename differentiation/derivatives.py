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
        self.bases = basis_vectors(self.var)
        self.nodes = [self.var]
        self.jac_external_shape = self.var.shape

    @property
    def jacobians(self):
        jac = {}
        for basis in self.bases:
            print(basis)
            self._partials[self.var.node_uid] = basis
            self._backwards_pass(basis)

        # reformat gradients into jacobians
        for node, grads in self.partials.items():
            if node in self.ind_vars:
                jac[node] = np.reshape(
                    self.partials[node],
                    self.jac_external_shape + self.var_internal_shapes[node]
                )

        return jac

    def _backwards_pass(self, basis):
        _partials = {
            self.var.node_uid: basis
        }

        for node in self.nodes:
            self.var_internal_shapes[node.node_uid] = node.shape

            if issubclass(node.__class__, UniTensorOperation):
                self.nodes.append(node.a)
                g_a = node.vector_jacobian_product(_partials[node.node_uid])

                if node.a not in _partials:
                    _partials[node.a.node_uid] = g_a
                else:
                    _partials[node.a.node_uid] += g_a

            elif issubclass(node.__class__, DualTensorOperation):
                self.nodes.append(node.a)
                self.nodes.append(node.b)
                g_a, g_b = node.vector_jacobian_product(_partials[node.node_uid])
                if node.a not in _partials:
                    _partials[node.a.node_uid] = g_a
                else:
                    _partials[node.a.node_uid] += g_a
                if node.b not in _partials:
                    _partials[node.b.node_uid] = g_b
                else:
                    _partials[node.b.node_uid] += g_b

            elif issubclass(node.__class__, Tensor):
                self.ind_vars.append(node.node_uid)

        for node in _partials:
            if node not in self.partials:
                self.partials[node] = [_partials[node]]
            else:
                self.partials[node] += [_partials[node]]
