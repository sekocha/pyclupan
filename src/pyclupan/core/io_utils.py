"""Utility functions for input/output."""


def write_list_no_space(a: list, file):
    """Write list without spaces.."""
    print("[", end="", file=file)
    print(*list(a), sep=",", end="]\n", file=file)
