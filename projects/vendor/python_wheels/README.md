# A81 offline dependency

Official PyPI wheel: `torchdiffeq-0.2.5-py3-none-any.whl` (32.9 kB).
Source and published checksum: https://pypi.org/project/torchdiffeq/0.2.5/#files
SHA256: `aa1db4bed13bd04952f28a53cdf4336d1ab60417c1d9698d7a239fec1cf2bcf8`.
Downloaded with certificate verification enabled, using pip's official PyPI index; bytes match the published checksum.

`run_a81_offline_20261104.sh` verifies this wheel and places it first on the child interpreter's PYTHONPATH.
The wheel contains pure Python and its original package metadata/license; Python imports it directly as a ZIP.
No server pip, package-index connection, certificate bypass, extraction, global installation or Torch/CUDA replacement is needed.
The same environment reaches the detached coordinator, mandatory tests and workers.

This repairs dependency acquisition only. The original A81 launcher, all SHA-bound experiment files, baseline,
policy, seeds, root, budgets, failure handling and results remain unchanged. It does not retry a failed model run.
