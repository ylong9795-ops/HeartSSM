"""Joint Shape64 / Motion32 / global SE(3) refinement contract.

All learned spatial conditions must reach the original decoder. Chamber offsets
are excluded. Optimization and acceptance policies remain to be designed.
"""


def optimize_joint(prior, observations, state):
    raise NotImplementedError('Joint Shape/Motion/global alignment optimization is not implemented')
