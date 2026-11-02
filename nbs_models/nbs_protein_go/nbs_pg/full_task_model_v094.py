"""One direct protein--GO scorer for both seen and unseen ontology rows.

Backbone/modelout predictions supervise training but are not inputs to this
readout. The v092 typed graph encoder and center/log-offset geometry path are
retained. Supplied new GO rows still require anchored ontology geometry; this
module neither generates BoxSquaredEL features nor certifies Stage1 zero-shot
provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

from .full_task_model_v092 import FullTaskGraphModelV092, FullTaskModelConfigV092
from .full_task_model_v093 import FullTaskGraphModelV093


@dataclass
class FullTaskModelConfigV094(FullTaskModelConfigV092):
    output_strategy: str = 'shared_direct'


class FullTaskGraphModelV094(FullTaskGraphModelV092):
    VERSION = '0.9.4'

    def __init__(self, protein_dim, ontology, config=None, *, seen_task_mask=None):
        cfg = config or FullTaskModelConfigV094()
        if cfg.output_strategy != 'shared_direct':
            raise ValueError('v094 requires output_strategy=shared_direct')
        super().__init__(protein_dim, ontology, cfg, seen_task_mask=seen_task_mask)
        self._ontology_revision = 0

    def forward(self, batch, *, target_ontology_ids=None, go_encoding=None,
                ontology_encoding=None, go_chunk_size=None, return_details=False):
        result = super().forward(batch, target_ontology_ids=target_ontology_ids,
                                 go_encoding=go_encoding,
                                 ontology_encoding=ontology_encoding,
                                 go_chunk_size=go_chunk_size,
                                 return_details=return_details)
        if return_details:
            # Compatibility with diagnostics: these are the SAME output and
            # shared parameters, not an auxiliary head or a second objective.
            result['shared_logits'] = result['logits']
        return result

    def replace_ontology_for_inference(self, updated_ontology, *, old_ontology_rows=None):
        # This validated resource operation only touches ontology buffers. It
        # does not instantiate the v093 model or its discarded residual head.
        result = FullTaskGraphModelV093.replace_ontology_for_inference(
            self, updated_ontology, old_ontology_rows=old_ontology_rows)
        result['schema'] = 'nbs-v094-anchored-ontology-replacement-v1'
        return result

    def representation_description(self):
        result = super().representation_description()
        result.update(output_strategy=self.config.output_strategy,
                      seen_GO='direct_shared_protein_GO_score',
                      unseen_GO='same_direct_shared_protein_GO_score',
                      backbone_logits_used_in_readout=False,
                      modelout_probabilities_used_in_readout=False,
                      rare_retention_objective=False,
                      trainable_GO_specific_output_rows=False)
        return result
