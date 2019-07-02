from variables.variables import Tensor, TensorConst


def is_tensor(x):
    return type(x) == Tensor or type(x) == TensorConst


def nest_func(external_func):
    def func_construction(internal_func):
        def wrapped_func(*args, **kwargs):
            g = external_func(*args, **kwargs)
            return internal_func(g)

        return wrapped_func
    return func_construction


