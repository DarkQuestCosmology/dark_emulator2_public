# dark_emulator2 公開登録手順

このファイルはローカル用メモ。GitHub には反映しない。

## PyPI 公開手順

### 1. 必要パッケージのインストール

```powershell
python -m pip install -U pip build twine
```

### 2. TestPyPI へアップロード

まず `pyproject.toml` の version を release candidate にする。

```toml
version = "1.0.0rc1"
```

同じ version は PyPI/TestPyPI に再アップロードできないので、やり直す場合は `rc2`, `rc3` のように上げる。

古い build 出力を消して build する。

```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info
python -m build
twine check dist/*
Get-ChildItem dist | Select-Object Name,Length
twine upload --repository testpypi dist/*
```

注意:

- PyPI の通常上限は upload file 1個あたり 100 MB。
- `dist/*.whl` と `dist/*.tar.gz` の両方が 100 MB 未満か確認する。
- 100 MB を超える場合は、PyPI の size limit increase 申請、または配布データの分離を検討する。

### 3. TestPyPI からインストール確認

新しい仮想環境で確認する。

```powershell
python -m venv .venv-testpypi
.\.venv-testpypi\Scripts\python -m pip install -U pip
.\.venv-testpypi\Scripts\python -m pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --pre dark_emulator2
```

`--pre` がないと release candidate が入らない。

最低限の確認:

```powershell
.\.venv-testpypi\Scripts\python -c "from dark_emulator2 import DarkEmulator2, __version__; print(__version__); de = DarkEmulator2(); print(de.param.get_fid_param()['Omega_m'])"
```

期待値:

```txt
1.0.0rc1
0.3156
```

### 4. PyPI へ正式アップロード

`pyproject.toml` の version から `rc` を外す。

```toml
version = "1.0.0"
```

古い build 出力を消して build する。

```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info
python -m build
twine check dist/*
Get-ChildItem dist | Select-Object Name,Length
twine upload --repository pypi dist/*
```

### 5. PyPI からインストール確認

新しい仮想環境で確認する。

```powershell
python -m venv .venv-pypi
.\.venv-pypi\Scripts\python -m pip install -U pip
.\.venv-pypi\Scripts\python -m pip install --no-cache-dir dark_emulator2
```

最低限の確認:

```powershell
.\.venv-pypi\Scripts\python -c "from dark_emulator2 import DarkEmulator2, __version__; print(__version__); de = DarkEmulator2(); print(de.param.get_fid_param()['Omega_m'])"
```

期待値:

```txt
1.0.0
0.3156
```

## Read the Docs 登録手順

### 1. GitHub 側の準備

GitHub の `DarkQuestCosmology/dark_emulator2_public` に以下を含めて push する。

- `.readthedocs.yaml`
- `docs/`
- `pyproject.toml`
- `README.md`
- `LICENSE`

Read the Docs Community で登録する場合、repository は public にする。

private repository のまま登録したい場合は Read the Docs for Business が必要。

### 2. Read the Docs に登録

1. Read the Docs にログインする。
2. GitHub 連携、または Read the Docs GitHub App を有効にする。
3. Dashboard で Add project を選ぶ。
4. `DarkQuestCosmology/dark_emulator2_public` を選ぶ。
5. project name は `dark-emulator2` など、URL として見やすい名前にする。
6. default branch を `main` にする。
7. build を実行する。

### 3. build log の確認

初回 build で見る点:

- `.readthedocs.yaml` が読まれているか。
- `docs/conf.py` が使われているか。
- dependency install で `torch` や `classy` が失敗していないか。
- API docs の autodoc import が通っているか。
- `debug` が API docs に出ていないか。

### 4. dependency install で失敗した場合

Read the Docs で `torch` や `classy` の install が重い、または失敗する場合は docs build 用に軽くする。

方針:

- package を docs build 用には `--no-deps` で install する。
- `docs/conf.py` に mock import を入れる。

候補:

```python
autodoc_mock_imports = ["torch", "classy"]
```

`.readthedocs.yaml` の候補:

```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"
  jobs:
    post_install:
      - python -m pip install --no-deps -e .

sphinx:
  configuration: docs/conf.py

python:
  install:
    - requirements: docs/requirements.txt
```

これは実際の build log を見てから判断する。

### 5. 公開 URL の確認

build 成功後、以下を確認する。

- top page が開く。
- Quickstart が開く。
- API Reference が開く。
- notebook link が `https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook` を指している。
- install command が `pip install dark_emulator2` になっている。
