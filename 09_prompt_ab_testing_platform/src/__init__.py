"""Prompt Versioning and A/B Testing Platform.

A database registry mapping prompt versions to configuration values and
developer change messages. Deterministic traffic distribution via fixed
hash user session splitting. Statistical winner declaration via scipy.stats
two-sample t-tests with automated kill switches.
"""

__version__ = "0.1.0"
