Automatic Differentiation
=========================

``dark_emulator2.autodiff`` returns the same outputs as ``DarkEmulator2`` (emulator method) as PyTorch tensors or JAX arrays,
differentiable with respect to the cosmological parameters.
See the notebooks ``08a_autodiff_torch.ipynb`` and ``08b_autodiff_jax.ipynb`` for examples.

.. currentmodule:: dark_emulator2.autodiff

.. autoclass:: TorchEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource

.. autoclass:: JaxEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource
