"""Narrow inference-only migration for pre-v0.7 checkpoints."""
from __future__ import annotations


def load_inference_state(model, state):
    """Strict load, allowing only the wholly absent, disabled GO residual branch.

    No trained tensor is changed and no active parameter may be missing. This
    does not migrate optimizer/scheduler state or authorize a training resume.
    """
    expected = model.state_dict()
    missing = set(expected) - set(state)
    prefixes = tuple('query_encoder.' + name for name in (
        'go_residual_query.', 'go_residual_key.', 'go_residual_fusion.', 'go_residual_gate.'))
    disabled_keys = {key for key in expected if key.startswith(prefixes)}
    added = []
    if missing and not model.config.use_go_residual_query and missing == disabled_keys:
        state = {**state, **{key: expected[key] for key in sorted(missing)}}
        added = sorted(missing)
    # Also rejects unexpected keys, other missing parameters and wrong shapes.
    model.load_state_dict(state, strict=True)
    return {'mode': 'disabled_go_residual_branch_only' if added else 'strict',
            'initialized_inactive_parameters': added}
