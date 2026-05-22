from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re


_DISTRIBUTION_NAME = "dark_emulator2"


def _version_from_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if match is None:
        return "unknown"
    return match.group(1)


try:
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = _version_from_pyproject()
