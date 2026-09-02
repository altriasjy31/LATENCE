from __future__ import annotations

from typing import Final, Tuple

EdgeType = Tuple[str, str, str]

PROTEIN: Final[str] = "protein"
GO: Final[str] = "go"

# Protein neighbourhood relations.
PPI: Final[EdgeType] = (PROTEIN, "ppi", PROTEIN)
SIMILAR_TO: Final[EdgeType] = (PROTEIN, "similar_to", PROTEIN)
WEAK_TO_CORE: Final[EdgeType] = (PROTEIN, "weak_to_core", PROTEIN)

# Protein--GO evidence is split by provenance.  Gold annotations, first-stage
# backbone candidates and expert-assisted pseudo annotations must never be
# collapsed into one generic annotation relation.
GOLD_PROTEIN_TO_GO: Final[EdgeType] = (PROTEIN, "gold_annotated_with", GO)
GOLD_GO_TO_PROTEIN: Final[EdgeType] = (GO, "has_gold_annotation", PROTEIN)

BACKBONE_CANDIDATE_PROTEIN_TO_GO: Final[EdgeType] = (
    PROTEIN,
    "backbone_rare_candidate",
    GO,
)
BACKBONE_CANDIDATE_GO_TO_PROTEIN: Final[EdgeType] = (
    GO,
    "candidate_of",
    PROTEIN,
)

PSEUDO_PROTEIN_TO_GO: Final[EdgeType] = (PROTEIN, "pseudo_annotated_with", GO)
PSEUDO_GO_TO_PROTEIN: Final[EdgeType] = (GO, "has_pseudo_annotation", PROTEIN)

# GO ontology directions.
GO_IS_A: Final[EdgeType] = (GO, "is_a", GO)            # child -> parent
GO_HAS_CHILD: Final[EdgeType] = (GO, "has_child", GO)  # parent -> child
GO_PART_OF: Final[EdgeType] = (GO, "part_of", GO)      # part -> whole
GO_HAS_PART: Final[EdgeType] = (GO, "has_part", GO)    # whole -> part

NBS_NODE_TYPES: Final[Tuple[str, ...]] = (PROTEIN, GO)
NBS_EDGE_TYPES: Final[Tuple[EdgeType, ...]] = (
    PPI,
    SIMILAR_TO,
    WEAK_TO_CORE,
    GOLD_PROTEIN_TO_GO,
    GOLD_GO_TO_PROTEIN,
    BACKBONE_CANDIDATE_PROTEIN_TO_GO,
    BACKBONE_CANDIDATE_GO_TO_PROTEIN,
    PSEUDO_PROTEIN_TO_GO,
    PSEUDO_GO_TO_PROTEIN,
    GO_IS_A,
    GO_HAS_CHILD,
    GO_PART_OF,
    GO_HAS_PART,
)
NBS_PROTEIN_GO_METADATA = (NBS_NODE_TYPES, NBS_EDGE_TYPES)

PROTEIN_GO_EVIDENCE_PAIRS: Final[Tuple[Tuple[EdgeType, EdgeType], ...]] = (
    (GOLD_PROTEIN_TO_GO, GOLD_GO_TO_PROTEIN),
    (BACKBONE_CANDIDATE_PROTEIN_TO_GO, BACKBONE_CANDIDATE_GO_TO_PROTEIN),
    (PSEUDO_PROTEIN_TO_GO, PSEUDO_GO_TO_PROTEIN),
)

# Compatibility aliases.  They intentionally resolve to the new gold relation
# names so old callers cannot silently reconstruct the obsolete generic schema.
TRUE_PROTEIN_TO_GO = GOLD_PROTEIN_TO_GO
TRUE_GO_TO_PROTEIN = GOLD_GO_TO_PROTEIN
PROTEIN_TO_GO = GOLD_PROTEIN_TO_GO
GO_TO_PROTEIN = GOLD_GO_TO_PROTEIN
PROTEIN_GO_METADATA = NBS_PROTEIN_GO_METADATA
