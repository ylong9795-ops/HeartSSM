"""Patient-wide rigid registration contract.

Estimate one canonical-to-patient SE(3), shared by four chambers and 24 phases,
against native 2D observations with explicit acquisition-plane poses. Global
candidate generation and selection remain to be implemented.
"""


def align_patient(prior, observations, state):
    raise NotImplementedError('Patient-wide global alignment is not implemented')
