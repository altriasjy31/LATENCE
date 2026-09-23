# NBS v0.5.6 — conservative background-PU supervision

This update keeps the v0.5.5 model, Q/C, ontology context, weak-focus sampler,
OneCycleLR and DDP contracts unchanged.  It only strengthens the negative/PU
side of each GO-query row.

## Motivation

In BP v0.5.5, only ~15% of GO queries have a backbone rare-candidate pool, so
many query rows contain gold/pseudo positives but no explicit unlabelled
contrast.  Unknown Protein–GO pairs must not be converted to closed-world
negatives, but a small conservative PU background can improve calibration of
the graph residual.

## New episode options

- `background_unlabelled_per_query`: default formal value 16.
- `background_unlabelled_weight`: default 0.05.
- `background_base_probability_max`: default 0.05.

A background pair is selected only from proteins already present in the shared
candidate union.  It must:

1. not already be a supervised pair in the row;
2. not belong to any gold annotation for this GO;
3. not belong to any modelout-positive pseudo annotation for this GO;
4. not belong to the full backbone-candidate pool for this GO;
5. have first-stage backbone probability <= the configured threshold.

The pair receives target 0 and low supervision weight 0.05.  It is therefore a
PU/background contrast signal, not an asserted biological negative.

No protein nodes, GO nodes or graph edges are added by this feature.

## New diagnostics

Per batch/epoch metadata now records:

- `background_pairs_requested/retained`
- `background_query_rows`
- `positive_query_rows`
- `negative_query_rows`
- `positive_only_query_rows`
- `negative_only_query_rows`
- `mixed_query_rows`
- positive/negative supervision pair counts
- negative-to-positive pair ratio

TQDM adds `bg`, `posonly`, `negrows`.
Epoch summary adds `bg`, `bg_fill`, `negrows`, `posonly`, `np_ratio`.

## Compatibility

The existing `gold_asl` metric name is preserved for checkpoint/log
compatibility.  As before, it denotes all non-pseudo masked ASL positions; in
v0.5.6 that set includes core gold positives, hard sampled-unlabelled pairs and
background-PU pairs, each with their own supervision weights.
