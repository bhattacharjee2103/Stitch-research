"""Auto-discovering registries for detectors, overlays and CLI commands.

Adding a detector, overlay or command means creating one new module in the
matching package and decorating an object with the matching decorator. No
registry or core file needs editing. See docs/ADDING.md.
"""
import importlib
import pkgutil


class Registry:
    """A named registry backed by the modules of one package."""

    def __init__(self, kind, package):
        self.kind = kind
        self.package = package
        self._items = {}

    def register(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind} name must be a non-empty string")

        def decorator(item):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = item
            return item

        return decorator

    def get(self, name):
        if name not in self._items:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self.kind} '{name}' (available: {available})")
        return self._items[name]

    def names(self):
        return sorted(self._items)

    def items(self):
        return dict(self._items)

    def discover(self):
        """Import every module in the package so its decorators run."""
        package = importlib.import_module(self.package)
        for module in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)
        return self


COMMANDS = Registry("command", "refcam.commands")
DETECTORS = Registry("detector", "refcam.detectors")
OVERLAYS = Registry("overlay", "refcam.overlays")


def register_command(name):
    return COMMANDS.register(name)


def register_detector(name):
    return DETECTORS.register(name)


def register_overlay(name):
    return OVERLAYS.register(name)


def load_commands():
    return COMMANDS.discover().items()


def load_detectors():
    return DETECTORS.discover().items()


def load_overlays():
    return OVERLAYS.discover().items()
