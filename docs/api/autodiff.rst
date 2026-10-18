Automatic Differentiation
=========================

With ``DarkEmulator2(autodiff="torch")`` or ``DarkEmulator2(autodiff="jax")``, ``get_pk``, ``get_lin_pk`` (and their aliases),
``get_cl_limber``, ``get_lin_cl_limber``, ``get_sigma8`` and ``get_lnAs`` with ``method="emulator"`` return PyTorch tensors or JAX arrays
that are differentiable with respect to the cosmological parameters; the other methods return numpy arrays as usual.
The computation is done by the classes below, which can also be used directly.

``device`` is ``"cpu"`` (default), ``"cuda"`` or ``"cuda:N"``. ``dtype`` follows the default of PyTorch or JAX.
Double precision is recommended for HMC and gradient checks; single precision is faster on GPUs.
See `torch.set_default_dtype <https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html>`_,
`JAX double precision <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`_
and `numpyro.enable_x64 <https://num.pyro.ai/en/stable/utilities.html#enable-x64>`_.
See the notebooks ``08a_autodiff_torch.ipynb`` and ``08b_autodiff_jax.ipynb`` for examples.

.. currentmodule:: dark_emulator2.autodiff

.. autoclass:: TorchEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource

.. autoclass:: JaxEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource
