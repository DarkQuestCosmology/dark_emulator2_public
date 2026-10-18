Quickstart
==========

Power spectra
-------------

.. code-block:: python

   from dark_emulator2 import DarkEmulator2

   de = DarkEmulator2()
   param = de.param.get_fid_param()

   k, pk = de.get_pk(param, zred=0.0)                        # non-linear, CDM+baryons
   k, pk_tot = de.get_pk_total(param, zred=[0.0, 0.5, 1.0])  # total matter, shape (3, len(k))
   k_lin, pk_lin = de.get_lin_pk(param, zred=0.0)            # linear, CDM+baryons

``k`` is in :math:`h\,\mathrm{Mpc}^{-1}` and :math:`P(k)` in :math:`(h^{-1}\,\mathrm{Mpc})^3`.
``zred`` can be a scalar or an array, and ``klist`` sets the output wavenumbers.

Cosmological parameters
-----------------------

A parameter dictionary needs ``ns``, ``w0``, ``wa`` and ``Mnu``, one of ``sigma8``, ``As`` and ``ln(10^10As)``,
three of ``Omega_m``, ``h0``, ``omega_b`` and ``omega_cdm``, and one of ``Omega_k`` and ``Omega_de``.
``set_param()`` checks the ranges and fills in the dependent parameters.

.. code-block:: python

   param = {"Omega_m": 0.315, "omega_b": 0.02225, "omega_cdm": 0.1198, "Mnu": 0.06, "Omega_k": 0.0,
            "sigma8": 0.831, "ns": 0.9645, "w0": -1.0, "wa": 0.0}
   param = de.param.set_param(param)
   de.is_in_support(param)  # inside the practical support of the emulator

Correlation function and lensing power spectrum
-----------------------------------------------

.. code-block:: python

   r, xi = de.get_xi(param, zred=0.0)
   ell, cl = de.get_cl_limber(param)  # Limber C(ell) for sources at z = 3

Automatic differentiation
-------------------------

With ``autodiff="torch"`` or ``autodiff="jax"``, the power spectra and ``C_ell`` are returned as PyTorch tensors or JAX arrays,
differentiable with respect to the parameters (see :doc:`api/autodiff`).

.. code-block:: python

   import torch

   emu = DarkEmulator2(autodiff="torch", device="cuda", dtype=torch.float64)  # or device="cpu"
   sigma8 = torch.tensor(0.831, dtype=torch.float64, requires_grad=True)
   k, pk = emu.get_pk({**param, "sigma8": sigma8}, zred=0.0)
   pk.sum().backward()  # d(sum P) / d sigma8 in sigma8.grad

More examples are in the `notebooks <https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook>`_.
