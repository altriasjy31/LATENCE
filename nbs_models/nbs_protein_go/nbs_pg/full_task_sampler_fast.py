"""Bounded-memory vectorization of the unchanged v088/v090 block sampler.

The supplied profile locates the hot loop inside ONE CPU batch construction;
it does not measure its share of end-to-end training. Pools are gathered in
receiver chunks (2048 by default) and filtered with array operations:

* eligible = not self, not forbidden/holdout, permitted, attr[0] > 0, first
  occurrence per receiver, in original pool order (the bounded refill);
* the fixed prefix and the dynamic draw use the SAME hash payload, generator,
  ``rng.choice`` call and weights, so every selected edge is identical;
* edges keep receiver-major, relation-code, eligible-position order; ``extra``
  sources are appended sorted; ``max_nodes`` raises under the same condition.

The original first-ACCEPTED-occurrence rule is retained, including skipping a
non-finite later duplicate only when that source was accepted earlier. On an
attribute or max_nodes error the original builder is rerun to preserve its
receiver/relation exception precedence. No sampling state is cached across
batches. Real-data equality and speed must be checked on the target server.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .full_task_data_v084 import RELATIONS
from .full_task_data_v088 import FullTaskDataV088
from .full_task_data_v090 import FullTaskDataV090
from .latence_graph_stores import RoleLocalProteinGOCSRStore

FAST_SAMPLER_VERSION = "2"


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _csr_rows(indptr, go_idx, rows, local):
    """Concatenated CSR rows as a [2, E] (local, GO) int64 edge array, row order kept."""
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return np.empty((2, 0), np.int64)
    starts = np.asarray(indptr[rows], np.int64)
    lengths = np.asarray(indptr[rows + 1], np.int64) - starts
    total = int(lengths.sum())
    if not total:
        return np.empty((2, 0), np.int64)
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(total)
    go = np.asarray(go_idx[offsets], np.int64)
    return np.stack((np.repeat(np.asarray(local, np.int64), lengths), go), axis=1).reshape(-1, 2).T


class FastBlockSamplerMixin:
    """Drop-in for FullTaskDataV088-derived readers; no new state or contract field."""

    # This is an execution/memory setting, never a sampling hyperparameter.
    # Both pool width and selected-edge storage still scale with the input.
    SAMPLER_RECEIVER_CHUNK = 2048

    def _forbidden_mask(self, forbidden):
        mask = np.zeros(self.registry.num_proteins, dtype=bool)
        ids = np.fromiter((int(x) for x in forbidden), dtype=np.int64, count=len(forbidden))
        ids = ids[(ids >= 0) & (ids < len(mask))]
        mask[ids] = True
        return mask

    def _layer_pools(self, current, relation, ghost_rows, ghost_neighbors, ghost_attrs):
        """[n, P] neighbour IDs (-1 = none) and [n, P, 3] attributes per receiver."""
        width = self.retrieval_pool_size if relation == "cosine" else self.pool_size
        ids = np.full((len(current), width), -1, np.int64)
        # Production pools are float32. Preserve input precision until after
        # eligibility nevertheless: the reference validates before copying to
        # float32, so converting here would change tiny/large float64 inputs.
        attr_dtype = (np.result_type(self._retrieval_attrs.dtype, ghost_attrs.dtype)
                      if relation == "cosine" else self._native[RELATIONS.index(relation)][2].dtype)
        attrs = np.zeros((len(current), width, 3), attr_dtype)
        known = np.flatnonzero(current >= 0)
        if relation == "cosine":
            if len(known):
                order = known[np.argsort(current[known], kind="stable")]  # sequential mmap reads
                ids[order] = np.asarray(self._retrieval_neighbors[current[order]], np.int64)
                attrs[order] = np.asarray(self._retrieval_attrs[current[order]])
            ghosts = np.flatnonzero(current < 0)
            if len(ghosts):
                rows = ghost_rows(current[ghosts])
                ids[ghosts] = np.asarray(ghost_neighbors[rows], np.int64)
                attrs[ghosts] = np.asarray(ghost_attrs[rows])
        elif len(known):
            lookup, neighbors, values = self._native[RELATIONS.index(relation)]
            rows = np.asarray(lookup[current[known]], np.int64)
            present = rows >= 0
            if present.any():
                target = known[present]
                order = np.argsort(rows[present], kind="stable")  # sequential mmap reads
                ids[target[order]] = np.asarray(neighbors[rows[present][order]], np.int64)
                attrs[target[order]] = np.asarray(values[rows[present][order]])
        return ids, attrs

    def _eligible(self, current, ids, attrs, forbidden_mask):
        """Original filter chain as masks; returns (eligible, eligible position)."""
        n_proteins = self.registry.num_proteins
        in_range = (ids >= 0) & (ids < n_proteins)
        safe = np.where(in_range, ids, 0)
        identity = (in_range & (ids != current[:, None]) & ~forbidden_mask[safe]
                    & self._permitted[safe])
        finite = np.isfinite(attrs).all(axis=2)
        positive = identity & finite & (attrs[..., 0] > 0)
        # Stable grouping puts equal IDs together while retaining pool order.
        # A zero-confidence first occurrence does NOT enter the original seen
        # set; a later non-finite occurrence must therefore still raise.
        width = ids.shape[1]
        key = np.where(identity, ids, -2 - np.arange(width, dtype=np.int64)[None, :])
        order = np.argsort(key, axis=1, kind="stable")
        ordered = np.take_along_axis(key, order, axis=1)
        group_start = np.ones_like(identity)
        group_start[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        column = np.arange(width, dtype=np.int64)[None, :]
        start = np.maximum.accumulate(np.where(group_start, column, 0), axis=1)
        ordered_positive = np.take_along_axis(positive, order, axis=1)
        last_positive = np.maximum.accumulate(np.where(ordered_positive, column, -1), axis=1)
        previous_positive = np.full_like(last_positive, -1)
        previous_positive[:, 1:] = last_positive[:, :-1]
        unseen = previous_positive < start
        invalid = np.take_along_axis(identity & ~finite, order, axis=1)
        if np.any(invalid & unseen):
            raise ValueError("v088 candidate edge attributes must be finite")
        valid = np.zeros_like(identity)
        np.put_along_axis(valid, order, ordered_positive & unseen, axis=1)
        return valid, np.cumsum(valid, axis=1) - 1

    def _dynamic_indices(self, protein, layer, relation, count, take, probabilities):
        """The original stable prefix plus hash-seeded weighted tail draw."""
        stable = min(take, int(take * self.stable_fraction))
        payload = f"{self.seed}:{self._sampling_step}:{self._sampling_rank}:{protein}:{layer}:{relation}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))
        tail = np.arange(stable, count)
        p = probabilities.get(len(tail))
        if p is None:
            weights = 1 / np.sqrt(np.arange(len(tail), dtype=np.float64) + 1)
            p = probabilities[len(tail)] = weights / weights.sum()
        chosen = rng.choice(tail, size=take - stable, replace=False, p=p)
        return np.concatenate((np.arange(stable), np.sort(chosen)))

    def _sample_layer(self, current, layer, ghost_rows, ghost_neighbors, ghost_attrs,
                      forbidden_mask, probabilities):
        """All selected edges of one layer: (target, code, position, source, attr), new count."""
        dynamic = self.mode == "dynamic" and self._sampling_training
        parts, new_count = [], 0
        chunk_rows = self.SAMPLER_RECEIVER_CHUNK
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows <= 0:
            raise ValueError("SAMPLER_RECEIVER_CHUNK must be a positive integer")
        for start in range(0, len(current), int(chunk_rows)):
            receivers = current[start:start + int(chunk_rows)]
            for code, relation in enumerate(RELATIONS):
                budget = self.fanouts[layer][relation]
                if not budget:
                    continue
                ids, attrs = self._layer_pools(receivers, relation, ghost_rows, ghost_neighbors, ghost_attrs)
                valid, position = self._eligible(receivers, ids, attrs, forbidden_mask)
                count = valid.sum(axis=1)
                take = np.minimum(budget, count)
                selected = valid & (position < take[:, None])
                if dynamic:
                    for row in np.flatnonzero(count > take):
                        indices = self._dynamic_indices(int(receivers[row]), layer, relation,
                                                        int(count[row]), int(take[row]), probabilities)
                        new_count += int(np.count_nonzero(indices >= take[row]))
                        columns = np.flatnonzero(valid[row])[indices]
                        selected[row] = False
                        selected[row, columns] = True
                targets, columns = np.nonzero(selected)
                if len(targets):
                    parts.append((targets + start, np.full(len(targets), code, np.int64),
                                  position[targets, columns], ids[targets, columns], attrs[targets, columns]))
        if not parts:
            return (np.empty(0, np.int64),) * 4 + (np.empty((0, 3), np.float32),), new_count
        merged = [np.concatenate(values) for values in zip(*parts)]
        order = np.lexsort((merged[2], merged[1], merged[0]))
        return [values[order] for values in merged], new_count

    def _build_blocks(self, query_ids, retrieval, forbidden):
        try:
            return self._build_blocks_fast(query_ids, retrieval, forbidden)
        except (ValueError, RuntimeError):
            # The vectorized traversal is not receiver/relation-major while
            # validating a whole chunk. Replay the reference on error so an
            # earlier max_nodes error is not replaced by a later NaN error.
            # RNGs are receiver-local; neither path consumes global RNG state.
            return super()._build_blocks(query_ids, retrieval, forbidden)

    def _build_blocks_fast(self, query_ids, retrieval, forbidden):
        self._require_pools()
        query_ids = np.asarray(query_ids, np.int64)
        if not len(query_ids) or np.any(query_ids >= 0) or len(np.unique(query_ids)) != len(query_ids):
            raise ValueError("v088 query IDs must be nonempty unique negative ghost IDs")
        neighbors, attributes = retrieval
        if (neighbors.shape != (len(query_ids), self.retrieval_pool_size)
                or attributes.shape != (*neighbors.shape, 3)):
            raise ValueError("v088 ghost retrieval must match the immutable cosine pool shape")
        ghost_neighbors = np.asarray(neighbors)
        ghost_attrs = np.asarray(attributes)
        ghost_order = np.argsort(query_ids, kind="stable")
        sorted_queries = query_ids[ghost_order]

        def ghost_rows(values):
            return ghost_order[np.searchsorted(sorted_queries, values)]

        forbidden_mask = self._forbidden_mask(forbidden)
        probabilities = {}
        current, reversed_blocks, new_count = query_ids, [], 0
        for layer in reversed(range(self.NUM_LAYERS)):
            if len(current) > self.max_nodes:
                raise RuntimeError("v088 max_nodes exceeded before feature gathering; reduce microbatch/fanouts")
            (targets, kinds, _, sources, attrs), changed = self._sample_layer(
                current, layer, ghost_rows, ghost_neighbors, ghost_attrs, forbidden_mask, probabilities)
            new_count += changed
            extra = np.unique(sources[~np.isin(sources, current)])
            if len(current) + len(extra) > self.max_nodes:
                raise RuntimeError("v088 max_nodes exceeded before feature gathering; reduce microbatch/fanouts")
            all_sources = np.concatenate((current, extra))
            sorter = np.argsort(all_sources, kind="stable")
            ordered = all_sources[sorter]

            def positions(values):
                return sorter[np.searchsorted(ordered, values)]

            block = dict(
                src_global_ids=np.asarray(all_sources, np.int64),
                dst_global_ids=np.asarray(current, np.int64),
                dst_in_src=np.asarray(positions(current), np.int64),
                edge_index=np.stack((positions(sources), targets), axis=1).astype(np.int64).reshape(-1, 2).T,
                edge_type=np.asarray(kinds, np.int64),
                edge_attr=np.asarray(attrs, np.float32).reshape(-1, 3),
            )
            reversed_blocks.append(block)
            current = all_sources
        blocks = list(reversed(reversed_blocks))
        for left, right in zip(blocks, blocks[1:]):
            if not np.array_equal(left["dst_global_ids"], right["src_global_ids"]):
                raise AssertionError("v088 block destination/source chain is inconsistent")
        return blocks, new_count

    def _label_edges(self, nodes, excluded):
        nodes = np.asarray(nodes, np.int64)
        known = nodes >= 0
        safe = np.where(known, nodes, 0)
        keep = known & ~self._forbidden_mask(excluded)[safe] & self._permitted[safe]
        codes = self.registry.role_code[safe]
        core_code = self.registry.role_to_code.get("core", -1)
        weak_code = self.registry.role_to_code.get("weak", -1)
        core = np.flatnonzero(keep & (codes == core_code))
        weak = np.flatnonzero(keep & (codes == weak_code))
        gold_store = self.stores.gold_messages
        gold = _csr_rows(gold_store.indptr, gold_store.go_idx, nodes[core], core)
        pseudo_store = self.stores.pseudo_messages
        rows = np.asarray(self.registry.role_row[nodes[weak]], np.int64)
        if isinstance(pseudo_store, RoleLocalProteinGOCSRStore):
            if len(rows) and (rows.min() < 0 or rows.max() >= pseudo_store.num_role_rows):
                raise IndexError("role-local protein row outside the pseudo CSR")
            pseudo = _csr_rows(pseudo_store.indptr, pseudo_store.go_idx, rows, weak)
        else:  # unknown store type: keep its own accessor semantics
            pairs = [(int(local), int(go)) for local, row in zip(weak, rows)
                     for go in pseudo_store.get_role_row(int(row))["go_idx"]]
            pseudo = np.asarray(pairs, np.int64).reshape(-1, 2).T
        return gold, pseudo


class FullTaskDataV088Fast(FastBlockSamplerMixin, FullTaskDataV088):
    """v088 reader with the vectorized sampler; VERSION and data contract unchanged."""


class FullTaskDataV090Fast(FastBlockSamplerMixin, FullTaskDataV090):
    """v090 reader (variable depth, label split) with the vectorized sampler."""
