"""
ipi.data — runtime data files shipped with the package.

This is a *data* directory, not a code module: it exists as a package only so
setuptools' ``package-data`` glob ships the JSON into the wheel (Kaggle installs
via ``pip install git+...``, which drops non-``.py`` files otherwise).

Contents:
  dual_verifiable_dataset.json — built by ``scripts/build_dual_verifiable_dataset.py``,
                                 loaded by ``ipi.dataset.DualVerifiableDataset``.
"""
