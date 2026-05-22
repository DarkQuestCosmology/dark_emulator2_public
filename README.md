# dark_emulator2

Matter power spectrum emulator in the non-flat $\nu w_0 w_a \mathrm{CDM}$ cosmology model.

## Installation

From PyPI:

```console
pip install dark_emulator2
```

From GitHub:

```console
pip install "git+https://github.com/DarkQuestCosmology/dark_emulator2_public.git"
```

From source:

```console
git clone https://github.com/DarkQuestCosmology/dark_emulator2_public.git
cd dark_emulator2_public
pip install .
```

- When building `classy` as a dependency, you may encounter an error indicating that `cython` is required. Since this dependency cannot be resolved by this package itself, please install them manually beforehand:

```console
pip install cython
pip install classy
```


## Usage

```python
from dark_emulator2 import DarkEmulator2

de = DarkEmulator2()
param = de.param.get_fid_param()
k, pk = de.get_pk(param, zred=0.0)
```

## Examples

Example notebooks are available in [DarkQuestCosmology/dark_emulator2_public](https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook).

## Reference

- matter power spectrum emulator
- Ginkaku code

## License

This project is distributed under the MIT License. See `LICENSE`.
