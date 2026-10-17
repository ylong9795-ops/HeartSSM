"""Inspect only the new state schema; old optimizer states are not migrated."""
from .io import load_state
from .state import TTOState


def validate_state(path, *, identity=None):
    state = TTOState.from_payload(load_state(path, identity=identity))
    return dict(patient_id=state.patient_id, stage=state.stage.value,
                schema_valid=True, identity=state.identity,
                diagnostics=state.diagnostics,
                interpretation='State contract only; no optimization or geometry success implied')
