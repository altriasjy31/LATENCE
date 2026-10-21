# NBS v0.5.6.1 — query-row metrics hotfix

This hotfix changes diagnostics only. It does not change model parameters, sampling, loss, background-PU selection, Q/C, DDP, ontology context, or OneCycleLR.

## Fix

v0.5.6 could print a contradictory epoch summary such as:

- live batch: `posonly=1, negrows=63` for `Q=64`
- epoch summary: `negrows=100.0% posonly=99.5%`

The live batch statistics were correct. The epoch percentage calculation reused averaged batch fields instead of computing rates from exact DDP-global query-row occurrence counts.

v0.5.6.1 now computes all row-balance rates using:

- global `positive_query_rows` occurrences
- global `negative_query_rows` occurrences
- global `positive_only_query_rows` occurrences
- global `negative_only_query_rows` occurrences
- global `mixed_query_rows` occurrences
- global `query_occurrences_total` denominator

It also records `query_row_partition_ok`, verifying:

`positive_only + mixed == positive_rows`

and

`negative_only + mixed == negative_rows`.

Epoch logs now show exact counts, e.g.:

`negrows=99.5%[20823/20928] posonly=0.5%[105/20928] mixed=99.5% rowpart=ok`

(the numbers above are illustrative).
