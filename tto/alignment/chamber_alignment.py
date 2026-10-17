"""Chamber translations produce INITIAL CANDIDATES ONLY (2026-10-17 decision).

Candidates must be fitted back into the original 64D Shape / 32D Motion decoder
and audited for inter-chamber anatomy before selection. Offsets may never become
saved final-mesh corrections or be applied after motion decoding. The candidate
algorithm and its anatomical acceptance rules remain to be implemented.
"""
MODE = 'initial_candidates_only'


def generate_candidates(prior, observations, state):
    raise NotImplementedError('Chamber initialization candidate generation is not implemented')
