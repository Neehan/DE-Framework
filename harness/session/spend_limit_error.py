"""Distinguish exhausted credentials from temporary provider cooldowns."""


class SpendLimitError(RuntimeError):
    """Stop the worker so the host disables its exhausted credential and selects another."""
