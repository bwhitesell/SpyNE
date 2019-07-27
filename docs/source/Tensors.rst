
==================
Tensors & Constants
===================

SpyNE.Tensor
============

.. class:: spyne.Tensor
    
    This is the fundamental data-structure of SpyNE. Under the hood, 
    the :obj:`Tensor` class is just a wrapper around numpy's :obj:`np.ndarray`.
       

        **Parameters:**


        :md_array:
        
            An object of type :obj:`np.ndarray` or a :obj:`list` of lists.
            Both options provide the necessary data and structure to define a 
            multi-dimensional array.

        :name:

            An optional object of type :obj:`str` that acts as a unique identifier
            for the tensor, primarily used by the :class:`spyne.gradients.BackwardsPass`
            class.



        Example: ::
            
            import spyne

            A = spyne.Tensor([[1, 2], [3, 4]])

            # or 
            import numpy as np
            A = spyne.Tensor(np.array([[1, 2], [3, 4]]))
        ::

