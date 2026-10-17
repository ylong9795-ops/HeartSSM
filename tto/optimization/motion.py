"""Optimize original 32D Motion through the original recursive decoder.

Pure motion evaluation lives in geometry.motion_metrics and does not constrain
the patient search to the initially predicted motion fields.
"""


def optimize_motion(prior, observations, state):
    raise NotImplementedError('32D Motion optimization is not implemented')
