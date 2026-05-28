# Dark Emulator2

Neural-network emulator for the matter power spectrum in the non-flat $\nu w_0 w_a o \mathrm{CDM}$ cosmology model, developed as part of the [Dark Quest project](https://darkquestcosmology.github.io/).

## Installation

From PyPI:

```console
pip install dark_emulator2
```

From GitHub:

```console
pip install git+https://github.com/DarkQuestCosmology/dark_emulator2_public.git
```

From source:

```console
git clone https://github.com/DarkQuestCosmology/dark_emulator2_public.git
cd dark_emulator2_public
pip install .
```

## Usage

```python
from dark_emulator2 import DarkEmulator2

de = DarkEmulator2()
param = de.param.get_fid_param()
k, pk = de.get_pk(param, zred=0.0)
```

## Examples

Example notebooks are available in [notebook](https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook).

## Documentation

API reference is available at [readthedocs](https://dark-emulator2.readthedocs.io/).

## Links

- Dark Emulator2 paper: [arXiv:2605.28596](https://doi.org/10.48550/arXiv.2605.28596)
- GINKAKU code paper: [arXiv:2605.28581](https://doi.org/10.48550/arXiv.2605.28581)

- Dark Quest project: [darkquestcosmology.github.io](https://darkquestcosmology.github.io/)
- Previous Dark Emulator release: [DarkQuestCosmology/dark_emulator_public](https://github.com/DarkQuestCosmology/dark_emulator_public)

- Supplementary emulator comparisons: https://darkquestcosmology.github.io/dark_emulator2_supplement/


## License

This project is distributed under the MIT License. See `LICENSE`.
