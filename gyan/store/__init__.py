from ..config import Settings
from .base import Store


def get_store(s: Settings) -> Store:
    if s.store == "monkdb":
        from .monkdb import MonkDBStore

        return MonkDBStore(s)
    from .local import LocalStore

    return LocalStore(s.db_path, dim=s.embed_dim)


__all__ = ["Store", "get_store"]
