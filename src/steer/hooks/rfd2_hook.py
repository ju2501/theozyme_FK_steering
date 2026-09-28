"""Explicit boundary: a real RFdiffusion2 adapter has NOT been implemented."""


class RFdiffusion2Sampler:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RFdiffusion2 sampler integration is not implemented or GPU-validated. "
            "Run scripts/probe_runtime.py and follow docs/RFD2_INTEGRATION.md. "
            "The toy demo is not RFdiffusion2."
        )
