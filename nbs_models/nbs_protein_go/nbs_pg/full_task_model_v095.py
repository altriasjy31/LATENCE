"""Heterogeneous tuned encoder and shared scorer, with seen hierarchy metadata."""
from dataclasses import dataclass

from .full_task_model_v094 import FullTaskGraphModelV094, FullTaskModelConfigV094
from .full_task_loss_v095 import build_hierarchy_task_edges


@dataclass
class FullTaskModelConfigV095(FullTaskModelConfigV094):
    jk_mode: str = 'sum'


class FullTaskGraphModelV095(FullTaskGraphModelV094):
    VERSION = '0.9.5'

    def __init__(self, protein_dim, ontology, config=None, *, seen_task_mask=None):
        super().__init__(protein_dim, ontology, config or FullTaskModelConfigV095(),
                         seen_task_mask=seen_task_mask)
        # Derived from the current immutable canonical axes, not trainable or
        # checkpointed state. No unseen endpoint receives hierarchy supervision.
        self.register_buffer('hierarchy_task_edges',
                             build_hierarchy_task_edges(ontology, self.seen_task_mask),
                             persistent=False)

    def replace_ontology_for_inference(self, updated_ontology, *, old_ontology_rows=None):
        result = super().replace_ontology_for_inference(
            updated_ontology, old_ontology_rows=old_ontology_rows)
        result['schema'] = 'nbs-v095-anchored-ontology-replacement-v1'
        return result
