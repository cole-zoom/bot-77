"""The scripted deck opponents (bot77.rl.deck_bots), one module per deck in docs/decks/. Importing this package loads
every module in it, which registers each deck in bot77.rl.deck_bots.DECKS."""
import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")
