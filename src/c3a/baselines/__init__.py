"""C3A ベースライン.

PAIR、Direct Request などの比較対象を提供する。
"""

from src.c3a.baselines.direct import (
    DirectBaseline,
    DirectConfig,
    create_direct_baseline,
)
from src.c3a.baselines.no_knowledge import (
    C3ANoKBBaseline,
    C3ANoKBConfig,
    create_c3a_nokb_baseline,
)
from src.c3a.baselines.pair import PAIRBaseline, PAIRConfig, create_pair_baseline

__all__ = [
    # PAIR
    "PAIRBaseline",
    "PAIRConfig",
    "create_pair_baseline",
    # Direct Request
    "DirectBaseline",
    "DirectConfig",
    "create_direct_baseline",
    # C3A-NoKB
    "C3ANoKBBaseline",
    "C3ANoKBConfig",
    "create_c3a_nokb_baseline",
]
