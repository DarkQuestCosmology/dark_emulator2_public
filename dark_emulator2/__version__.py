try:
    from importlib.metadata import version
    __version__ = version("dark_emulator2")
except Exception:
    __version__ = "unknown"
