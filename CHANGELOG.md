# Changelog

## 1.1.0 - 2026-10-18

- Add `dark_emulator2.autodiff`: `TorchEmulator` and `JaxEmulator` return the linear/nonlinear `P(k)`, `sigma8`/`ln(10^10As)` and Limber `C_ell` of `DarkEmulator2` (emulator method) as PyTorch tensors or JAX arrays, differentiable with respect to the cosmological parameters (e.g. for HMC with Pyro or NumPyro). Add `jax` and `array-api-compat` to the dependencies.
- Add notebooks `08a_autodiff_torch.ipynb` and `08b_autodiff_jax.ipynb`.
- Store the network weights as plain `state_dict` files read with `torch.load(weights_only=True)` (`dark_emulator2.network`) instead of TorchScript, which is deprecated in recent PyTorch; the weights and outputs are unchanged.
- Ship only the data files read by the code (the parameter lists and unused fiducial spectra stay in the repository); the wheel shrinks from 68 MB to 64 MB.
- Fix the inverted h-unit conversion of `k` and `P(k)` in `pk2cl_limber()` and `pk2cl_limber_src_dist()`, which affected all Limber `C_ell` outputs (`get_cl_limber()`, `get_lin_cl_limber()`).
- Fix the source-averaged lensing kernel in `pk2cl_limber_src_dist()` used with `src_dist`.
- Fix the divergence of `get_xi()` and `get_lin_xi()` when the `P(k)` tail is shallower than about `k^-2` (e.g. at high redshift).
- Fix the arguments passed to FFTLog in `fftlog.py`; `xi(r)` now has 600 points starting at `r = 1.02e-3`.
- Use `Omega_de = 1 - Omega_m - Omega_k` in `comoving_distance()`.
- Fix the handling of `As` and `ln(10^10As)` in `method="class"`, `get_As()`, `get_sigma8()` and `is_in_support()`; `get_sigma8()` raises `ValueError` instead of exiting.
- Return outputs in the input order for non-monotonic `zred`, and restore `reduce_shotnoise` when a call inside `get_xi()` or `get_*cl_limber()` fails.
- Minor fixes in the shot-noise extrapolation, `tophat_nsz()`, `set_params_from_file_to_dict()` and `get_preset_range_dict_pair()`.
- Rerun notebooks `03a`, `03b`, `06b`, `07a` and `07b` with these fixes; tidy notebooks `00`-`04` and expand the quickstart page of the documentation.

## 1.0.2 - 2026-09-24

- Fix `comoving_distance()` to account for CPL dark energy evolution (`w0`, `wa`), while retaining ΛCDM defaults when those parameters are absent.

## 1.0.1 - 2026-05-27

- Add DE2 support-check helpers: `DarkEmulator2.is_in_support()` and `Parameters.get_de2_om_sig8_plane()`.
- Update public example notebooks with DE2 support examples.

## 1.0.0 - 2026-05-22

- Use `pyproject.toml` as the single source of truth for the package version.
- Add `LICENSE` and move the historical version notes from `dark_emulator2/__version__.py` into this changelog.

## 0.2.4.dev0 - 2025-12-01

- Add massless neutrino w0waoCDM parameter generation.

## 0.2.4c - 2025-10-15

- Adjust dependency versions.

## 0.2.4b - 2025-07-10

- Fix bug in `set_param` input parameter consistency.

## 0.2.4a - 2025-07-09

- Add strict parameter range option.

## 0.2.4 - 2025-07-06

- Add docstrings.

## 0.2.3 - 2025-07-05

- Change the file structures.

## 0.2.2 - 2025-07-03

- Update parameter class.

## 0.2.1 - 2025-07-02

- Update README and support verbose mode.

## 0.2.0 - 2025-07-01

- Update network models.

## 0.1.2 - 2024-01-24

- Add fftlog functions.

## 0.1.1 - 2024-01-18

- Optimize shot noise removal.

## 0.1.0 - 2023-12-07

- Support test public version.

## 0.0.6a - 2023-12-04

- Stop supporting GPU estimation.

## 0.0.6 - 2023-11-08

- Add middle-resolution random seed models.
- Support GPU estimation.

## 0.0.5 - 2023-10-09

- Add high-resolution models for fixed model.
- Add reduce shot noise option.

## 0.0.4a - 2023-09-02

- Change internal structures.

## 0.0.4 - 2023-08-31

- Support z-list input.
- Avoid scipy `RegularGridInterpolator` bug.

## 0.0.3 - 2023-08-01

- Support total matter estimation.

## 0.0.2 - 2023-07-26

- Change random parameter generator.

## 0.0.1 - 2023-07-05

- First commit.
