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

## Links

- dark_emulator2 paper: arxiv.xxx
- GINKAKU code paper: arxiv.xxx

- Dark Quest project: [darkquestcosmology.github.io](https://darkquestcosmology.github.io/)
- Previous Dark Emulator release: [DarkQuestCosmology/dark_emulator_public](https://github.com/DarkQuestCosmology/dark_emulator_public)

- Supplementary emulator comparisons: https://darkquestcosmology.github.io/dark_emulator2_supplement/


## License

This project is distributed under the MIT License. See `LICENSE`.
