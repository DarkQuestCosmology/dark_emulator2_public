Automatic Differentiation
=========================

With ``DarkEmulator2(autodiff="torch")`` or ``DarkEmulator2(autodiff="jax")``, ``get_pk``, ``get_lin_pk`` (and their aliases),
``get_cl_limber``, ``get_lin_cl_limber`` and ``get_sigma8`` with ``method="emulator"`` return PyTorch tensors or JAX arrays
that are differentiable with respect to the cosmological parameters. ``get_lnAs`` also supports automatic differentiation.
The other methods return numpy values as usual.
The computation is done by the classes below, which can also be used directly.

``device`` is ``"cpu"`` (default), ``"cuda"`` or ``"cuda:N"``. ``dtype`` follows the default of PyTorch or JAX.
Double precision is recommended for HMC and gradient checks; single precision is faster on GPUs.
See `torch.set_default_dtype <https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html>`_,
`JAX double precision <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`_
and `numpyro.enable_x64 <https://num.pyro.ai/en/stable/utilities.html#enable-x64>`_.
See the notebooks ``08a_autodiff_torch.ipynb`` and ``08b_autodiff_jax.ipynb`` for examples.

Parameter inputs
----------------

Pass the eight independent parameters ``Omega_m``, ``omega_b``, ``ns``, ``Mnu``, ``Omega_k``, ``w0``, ``wa`` and ``h0``,
and one amplitude parameter: ``sigma8``, ``As`` or ``ln(10^10As)``.
The amplitude is converted to ``sigma8`` within the differentiable calculation when needed.
Use exactly one amplitude parameter when constructing an input dictionary; if several are present, a non-None ``sigma8``
takes precedence, followed by ``ln(10^10As)`` and then ``As``.

Build the dictionary directly from the tensors or JAX arrays being differentiated.
Use ``emu.param.set_param()`` only to prepare ordinary numerical parameters before the differentiable calculation.
It copies its inputs and uses NumPy/SciPy; calling it inside a gradient function can disconnect PyTorch inputs
from the result or fail with JAX tracers.
Dependent entries such as ``omega_cdm`` and ``Omega_de`` in a completed dictionary do not need to be updated for these calls:
the differentiable calculation uses the independent parameters listed above.

For example, this computes the derivative of the sum of P(k) with respect to ``sigma8`` on the CPU:

.. code-block:: python

   import torch
   from dark_emulator2 import DarkEmulator2

   emu = DarkEmulator2(autodiff="torch", dtype=torch.float64)
   param = emu.param.get_fid_param()
   sigma8 = torch.tensor(0.831, dtype=torch.float64, requires_grad=True)
   k, pk = emu.get_pk({**param, "sigma8": sigma8})
   pk.sum().backward()
   print(sigma8.grad)

To vary ``As`` instead, first remove ``sigma8`` and ``ln(10^10As)`` from ``param``.
With autodiff enabled, ``get_sigma8()`` and ``get_lnAs()`` return their result without updating the input dictionary;
assign the returned value when it is needed for another calculation.

Grids and emulator settings
---------------------------

Automatic differentiation is with respect to cosmological parameters.
Keep ``zred``, ``klist``, ``zlist``, ``llist`` and ``src_dist`` fixed; use NumPy arrays for array-valued grids and source distributions.
P(k) has shape ``(batch..., nz, nk)``, with the redshift axis dropped for scalar ``zred``.
C(ell) has shape ``(batch..., nl)``. Parameter arrays must have compatible broadcast shapes.

Set emulator options and grids before compiling a function with ``jax.jit``.
If they change, create a new function and compile it again so that JAX captures the new settings.
Outside a compiled function, changes to the linear-spectrum connection and shot-noise settings take effect on the next call.
As in the ordinary emulator, C(ell) always uses the shot-noise tail treatment;
its derivatives are local to the selected cutoff and can change when the cutoff switches.

Emulator classes
----------------

.. currentmodule:: dark_emulator2.autodiff

.. autoclass:: TorchEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource

.. autoclass:: JaxEmulator
   :members: get_sigma8, get_lnAs, get_lin_pk, get_pk, get_lin_cl_limber, get_cl_limber
   :member-order: bysource
