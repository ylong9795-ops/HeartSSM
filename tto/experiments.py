"""Stage capability registry, consumed only by tto.run and the state machine."""
CAPABILITIES = {
    'preflight': 'implemented: configuration and native observation contracts; no model load',
    'initialize': 'implemented: exact frozen original forward and versioned state',
    'global_alignment': 'implemented: canonical-gauge native centre/depth/tilt plane candidates; frozen meshes',
    'chamber_candidates': 'implemented S2: observable relative hints -> original Regional8; fixed Global32/Motion32/Pose; hard geometry admission',
    'shape': 'not implemented: original Shape64 search',
    'motion': 'not implemented: original Motion32 search',
    'joint': 'not implemented: Shape/Motion/global SE(3) refinement',
    'validate': 'implemented: inspect a saved state, not clinical or optimization validation',
}
