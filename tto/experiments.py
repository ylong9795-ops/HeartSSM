"""Stage capability registry, consumed only by tto.run and the state machine."""
CAPABILITIES = {
    'preflight': 'implemented: configuration and native observation contracts; no model load',
    'initialize': 'implemented: exact frozen original forward and versioned state',
    'global_alignment': 'not implemented: shared patient SE(3) registration',
    'chamber_candidates': 'not implemented: initialization candidates only',
    'shape': 'not implemented: original Shape64 search',
    'motion': 'not implemented: original Motion32 search',
    'joint': 'not implemented: Shape/Motion/global SE(3) refinement',
    'validate': 'implemented: inspect a saved state, not clinical or optimization validation',
}
