"""Prompt groups for HIS experiments.

Each group module exports two lists: ``public_prompts`` (short, generic) and
``personal_prompts`` (long, specific). Generators load a group by index.
"""

from typing import List, Tuple


def load_group(n: int) -> Tuple[List[str], List[str]]:
    if n == 1:
        from . import group1 as g
    elif n == 2:
        from . import group2 as g
    else:
        raise ValueError(f"unknown prompt group: {n}")
    return list(g.public_prompts), list(g.personal_prompts)
