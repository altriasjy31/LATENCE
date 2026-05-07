import collections as clt
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch


ONTO_TYPES = ("cc", "mf", "bp")

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES


def get_goplus_defs(filename='data/definitions.txt'):
    plus_defs = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            go_id, definition = line.split(': ')
            go_id = go_id.replace('_', ':')
            definition = definition.replace('_', ':')
            plus_defs[go_id] = set(definition.split(' and '))
    return plus_defs


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = clt.Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = clt.deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = clt.deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs


def read_go_terms(path: Union[str, Path]) -> List[str]:
    """
    读取 GO label 顺序文件。

    支持格式：
        GO:0005575
        GO:0005575<TAB>cellular_component
        GO:0005575 cellular_component
        GO:0005575,cellular_component

    只取第一列作为 GO ID。
    """
    terms = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            term = line.replace(",", "\t").split()[0]
            terms.append(term)
    return terms


def write_go_terms(path: Union[str, Path], terms: Sequence[str]) -> None:
    with open(path, "w") as f:
        for t in terms:
            f.write(f"{t}\n")


def get_all_namespace_canonical_terms(
    ont: Ontology,
    ont_type: str,
    include_root: bool = False,
) -> List[str]:
    """
    从 OBO 中提取某一 namespace 的 canonical GO terms。

    注意：
    你的 Ontology.load() 会把 alt_id 也作为 key 放进 ont。
    因此这里用 go_id == obj["id"] 来排除 alt_id 的重复项。
    """
    if ont_type not in NAMESPACES:
        raise ValueError(f"Unknown ontology type: {ont_type}")

    namespace = NAMESPACES[ont_type]
    root_id = FUNC_DICT[ont_type]

    terms = []

    for go_id, obj in ont.ont.items():
        if obj.get("is_obsolete", False):
            continue

        canonical_id = obj["id"]

        # 排除 alt_id key，避免同一个 term 出现多次
        if go_id != canonical_id:
            continue

        if obj.get("namespace") != namespace:
            continue

        if not include_root and canonical_id == root_id:
            continue

        terms.append(canonical_id)

    return sorted(terms)


def canonicalize_label_terms(
    ont: Ontology,
    terms: Sequence[str],
    ont_type: str,
    duplicate_policy: str = "error",
) -> List[str]:
    """
    将 label terms 统一为 canonical GO ID，同时保持顺序。

    duplicate_policy:
        "error":
            如果多个 label term 映射到同一个 canonical GO ID，则报错。
            这是最严格、最安全的模式。

        "keep":
            保留所有 label columns。
            例如 GO:0000776 和 GO:0031617 都映射到 GO:0000776，
            返回结果中会出现两个 GO:0000776。
            适合已有模型输出维度不能改的情况。

        "drop":
            只保留第一次出现的 canonical GO ID。
            注意：这会改变 label 数量，必须同步重建 target tensor 和模型输出层。
    """
    if duplicate_policy not in {"error", "keep", "drop"}:
        raise ValueError(
            f"duplicate_policy must be one of 'error', 'keep', 'drop', "
            f"got {duplicate_policy!r}"
        )

    if ont_type not in NAMESPACES:
        raise ValueError(f"Unknown ontology type: {ont_type}")

    namespace = NAMESPACES[ont_type]

    canonical_terms = []
    seen = {}

    for i, go_id in enumerate(terms):
        go_id = str(go_id).strip()

        obj = ont.get_term(go_id)

        if obj is None:
            raise ValueError(
                f"GO term {go_id} at position {i} is absent from ontology."
            )

        if obj.get("is_obsolete", False):
            raise ValueError(
                f"GO term {go_id} at position {i} is obsolete."
            )

        if obj.get("namespace") != namespace:
            raise ValueError(
                f"GO term {go_id} at position {i} belongs to namespace "
                f"{obj.get('namespace')}, but expected {namespace} for {ont_type}."
            )

        canonical_id = obj["id"]

        if canonical_id in seen:
            if duplicate_policy == "error":
                raise ValueError(
                    f"Duplicate canonical GO term after alt_id mapping: {canonical_id}. "
                    f"First seen as {seen[canonical_id]}, then as {go_id}. "
                    f"If you want to keep the current label columns, rerun with "
                    f"duplicate_policy='keep'. If you want a clean label vocabulary, "
                    f"use duplicate_policy='drop' and rebuild your targets/model head."
                )

            if duplicate_policy == "drop":
                continue

            # duplicate_policy == "keep"
            # 保留当前 column。
        else:
            seen[canonical_id] = go_id

        canonical_terms.append(canonical_id)

    return canonical_terms

def validate_label_terms_preserve_order(
    ont: Ontology,
    terms: Sequence[str],
    ont_type: str,
) -> Tuple[List[str], List[str], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    验证 label terms，但不 canonicalize 输出顺序，不去重，不重排。

    目的：
        保持已有训练数据的 label column 顺序不变。

    Returns
    -------
    label_terms_preserved:
        原始 label terms，顺序不变。
        第 i 个 term 仍然对应 logits[:, i]。

    idx_to_canonical:
        长度等于 label_terms_preserved。
        idx_to_canonical[i] 是 label_terms_preserved[i] 在 OBO 中对应的 canonical GO ID。

    canonical_to_indices:
        canonical GO ID -> 所有对应的 label column indices。
        例如：
            {
                "GO:0000776": [10, 25]
            }

    duplicate_canonical_groups:
        canonical 后有多个 label column 的 term。
        这不是错误，只是记录。
    """
    if ont_type not in NAMESPACES:
        raise ValueError(f"Unknown ontology type: {ont_type}")

    namespace = NAMESPACES[ont_type]

    label_terms_preserved = []
    idx_to_canonical = []
    canonical_to_indices = clt.defaultdict(list)

    for i, go_id in enumerate(terms):
        go_id = str(go_id).strip()

        if not go_id:
            raise ValueError(f"Empty GO term at position {i}.")

        obj = ont.get_term(go_id)

        if obj is None:
            raise ValueError(
                f"GO term {go_id} at position {i} is absent from ontology."
            )

        if obj.get("is_obsolete", False):
            raise ValueError(
                f"GO term {go_id} at position {i} is obsolete."
            )

        if obj.get("namespace") != namespace:
            raise ValueError(
                f"GO term {go_id} at position {i} belongs to namespace "
                f"{obj.get('namespace')}, but expected {namespace} for {ont_type}."
            )

        canonical_id = obj["id"]

        label_terms_preserved.append(go_id)
        idx_to_canonical.append(canonical_id)
        canonical_to_indices[canonical_id].append(i)

    canonical_to_indices = dict(canonical_to_indices)

    duplicate_canonical_groups = {
        canonical_id: indices
        for canonical_id, indices in canonical_to_indices.items()
        if len(indices) > 1
    }

    return (
        label_terms_preserved,
        idx_to_canonical,
        canonical_to_indices,
        duplicate_canonical_groups,
    )

def build_go_edges(
    ont: Ontology,
    label_terms: Sequence[str],
    ont_type: str,
    transitive: bool = False,
    duplicate_parent_mode: str = "representative",
    add_alias_equivalence_edges: bool = False,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    """
    构建某个 ontology 类型的 GO edges。

    重要：
        本函数保持 label_terms 的原始顺序和长度不变。
        不会因为 alt_id -> canonical_id 后重复而删除 label column。

    Parameters
    ----------
    ont:
        Ontology 对象。

    label_terms:
        模型该 head 的输出标签顺序。
        第 i 个 term 对应 logits[:, i]。
        本函数不会重排、去重或合并它。

    ont_type:
        "cc", "mf", or "bp"。

    transitive:
        False:
            只加入直接 child -> parent 边。
        True:
            加入 child -> 所有 ancestor 边。

    duplicate_parent_mode:
        当某个 parent canonical GO ID 在 label list 中对应多个 column 时如何处理。

        "representative":
            默认。只选择一个代表 parent column。
            优先选择 raw GO ID 正好等于 canonical ID 的 column；
            如果没有，则选择该 canonical group 的第一个 column。

            这是对已有训练数据最小侵入的方式。

        "all":
            child 同时约束到所有 parent alias columns。
            语义上更强，但如果你的训练标签矩阵中 alias columns 不一致，
            可能引入额外冲突。

    add_alias_equivalence_edges:
        是否为同一个 canonical GO ID 的多个 label columns 加双向边。

        例如：
            GO:0000776 和 GO:0031617 都映射到 GO:0000776

        如果 True，则加入：
            GO:0000776 -> GO:0031617
            GO:0031617 -> GO:0000776

        在你的 loss 下，这会近似强迫两个 alias 概率相等。
        但如果已有 target columns 不一致，可能和监督信号冲突。
        因此默认 False。

    Returns
    -------
    edges:
        torch.LongTensor, shape [E, 2]
        每行是 [child_index, parent_index]。

    label_terms_preserved:
        原始 label terms，顺序不变。
        与 logits 列顺序一致。

    stats:
        构建统计信息。
    """
    if ont_type not in NAMESPACES:
        raise ValueError(f"Unknown ontology type: {ont_type}")

    if duplicate_parent_mode not in {"representative", "all"}:
        raise ValueError(
            "duplicate_parent_mode must be either 'representative' or 'all', "
            f"got {duplicate_parent_mode}"
        )

    namespace = NAMESPACES[ont_type]

    (
        label_terms_preserved,
        idx_to_canonical,
        canonical_to_indices,
        duplicate_canonical_groups,
    ) = validate_label_terms_preserve_order(
        ont=ont,
        terms=label_terms,
        ont_type=ont_type,
    )

    # 为每个 canonical GO ID 选择一个代表 label index。
    #
    # 原则：
    #   1. 如果 canonical ID 本身就在 label list 中，优先用它；
    #   2. 否则使用这个 canonical group 的第一个出现位置。
    #
    # 这样可以避免因为 alt_id 重复导致 term_to_idx 覆盖。
    canonical_to_representative_idx = {}

    for canonical_id, indices in canonical_to_indices.items():
        exact_indices = [
            i for i in indices
            if label_terms_preserved[i] == canonical_id
        ]

        if exact_indices:
            representative_idx = exact_indices[0]
        else:
            representative_idx = indices[0]

        canonical_to_representative_idx[canonical_id] = representative_idx

    edge_set = set()

    parent_links_seen = 0
    candidate_edges_seen = 0
    skipped_not_in_labels = 0
    skipped_cross_namespace = 0
    skipped_missing_or_obsolete = 0
    skipped_same_canonical = 0

    for child_idx, child_raw_id in enumerate(label_terms_preserved):
        child_canonical_id = idx_to_canonical[child_idx]

        if transitive:
            # 用 canonical child 查询 ancestor，避免 alt_id 自身混入 ancestor set。
            parent_ids = set(ont.get_anchestors(child_canonical_id))
            parent_ids.discard(child_canonical_id)
        else:
            parent_ids = set(ont.get_parents(child_canonical_id))

        for parent_id in parent_ids:
            parent_links_seen += 1

            parent_obj = ont.get_term(parent_id)

            if parent_obj is None or parent_obj.get("is_obsolete", False):
                skipped_missing_or_obsolete += 1
                continue

            if parent_obj.get("namespace") != namespace:
                skipped_cross_namespace += 1
                continue

            parent_canonical_id = parent_obj["id"]

            # 正常 GO DAG 不应出现 child 和 parent canonical 相同。
            # 但如果 relationship 或 alt_id 造成这种情况，不应把 alias 当作层级边。
            if parent_canonical_id == child_canonical_id:
                skipped_same_canonical += 1
                continue

            if duplicate_parent_mode == "representative":
                parent_idx = canonical_to_representative_idx.get(parent_canonical_id)

                if parent_idx is None:
                    skipped_not_in_labels += 1
                    continue

                parent_indices = [parent_idx]

            else:
                parent_indices = canonical_to_indices.get(parent_canonical_id)

                if not parent_indices:
                    skipped_not_in_labels += 1
                    continue

            candidate_edges_seen += len(parent_indices)

            for parent_idx in parent_indices:
                if child_idx == parent_idx:
                    continue

                edge_set.add((child_idx, parent_idx))

    alias_equivalence_edges_added = 0

    if add_alias_equivalence_edges:
        for canonical_id, indices in duplicate_canonical_groups.items():
            if len(indices) <= 1:
                continue

            for i in indices:
                for j in indices:
                    if i == j:
                        continue
                    if (i, j) not in edge_set:
                        alias_equivalence_edges_added += 1
                    edge_set.add((i, j))

    if len(edge_set) == 0:
        edges = torch.empty((0, 2), dtype=torch.long)
    else:
        edges = torch.tensor(sorted(edge_set), dtype=torch.long)

    duplicate_examples = []

    for canonical_id, indices in sorted(duplicate_canonical_groups.items())[:20]:
        representative_idx = canonical_to_representative_idx[canonical_id]

        duplicate_examples.append(
            {
                "canonical_id": canonical_id,
                "indices": list(indices),
                "raw_terms": [label_terms_preserved[i] for i in indices],
                "representative_idx": representative_idx,
                "representative_term": label_terms_preserved[representative_idx],
            }
        )

    stats = {
        "num_labels": len(label_terms_preserved),
        "num_edges": int(edges.shape[0]),
        "parent_links_seen": parent_links_seen,
        "candidate_edges_seen": candidate_edges_seen,
        "skipped_not_in_labels": skipped_not_in_labels,
        "skipped_cross_namespace": skipped_cross_namespace,
        "skipped_missing_or_obsolete": skipped_missing_or_obsolete,
        "skipped_same_canonical": skipped_same_canonical,
        "transitive": int(transitive),
        "duplicate_parent_mode": duplicate_parent_mode,
        "add_alias_equivalence_edges": int(add_alias_equivalence_edges),
        "alias_equivalence_edges_added": alias_equivalence_edges_added,
        "num_duplicate_canonical_groups": len(duplicate_canonical_groups),
        "num_duplicate_label_columns": sum(
            len(indices) - 1
            for indices in duplicate_canonical_groups.values()
        ),
        "duplicate_canonical_examples": duplicate_examples,
    }

    return edges, label_terms_preserved, stats


def save_go_edges_for_all_namespaces(
    obo_path: Union[str, Path],
    out_dir: Union[str, Path],
    label_files: Optional[Mapping[str, Union[str, Path]]] = None,
    label_terms: Optional[Mapping[str, Sequence[str]]] = None,
    with_rels: bool = False,
    transitive: bool = False,
    include_root: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    分别构建并保存 cc / mf / bp 三种 ontology 的 edges。

    保存文件：
        out_dir/go_edges_cc.pt
        out_dir/go_edges_mf.pt
        out_dir/go_edges_bp.pt

    同时保存 canonicalized term 顺序：
        out_dir/go_terms_cc.txt
        out_dir/go_terms_mf.txt
        out_dir/go_terms_bp.txt

    Parameters
    ----------
    obo_path:
        go.obo 文件路径。
    out_dir:
        输出目录。
    label_files:
        可选。形如：
            {
                "cc": "terms_cc.txt",
                "mf": "terms_mf.txt",
                "bp": "terms_bp.txt",
            }

        如果提供，则必须保证这些 term 顺序和 logits 列顺序一致。

    label_terms:
        可选。直接传入 Python list：
            {
                "cc": [...],
                "mf": [...],
                "bp": [...],
            }

    with_rels:
        传给你的 Ontology(filename, with_rels=with_rels)。
        注意：你当前的 parser 在 with_rels=True 时会把所有 relationship
        都加入 is_a，包括 regulates 等关系。一般建议先用 False。

    transitive:
        是否加入所有 ancestor 边。
        如果 label set 不是完整 GO 闭包，建议 True。

    include_root:
        仅在 label_files 和 label_terms 都没有提供时有效。
        表示从 OBO 自动生成全量 terms 时是否包含三大根节点。

    Returns
    -------
    stats_by_type:
        每种 ontology 的统计信息。
    """
    if label_files is not None and label_terms is not None:
        raise ValueError("Use either label_files or label_terms, not both.")

    if label_files is not None:
        missing = set(ONTO_TYPES) - set(label_files.keys())
        if missing:
            raise ValueError(f"label_files missing keys: {sorted(missing)}")

    if label_terms is not None:
        missing = set(ONTO_TYPES) - set(label_terms.keys())
        if missing:
            raise ValueError(f"label_terms missing keys: {sorted(missing)}")

    ont = Ontology(filename=str(obo_path), with_rels=with_rels)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_by_type = {}

    for ont_type in ONTO_TYPES:
        if label_files is not None:
            terms = read_go_terms(label_files[ont_type])
        elif label_terms is not None:
            terms = list(label_terms[ont_type])
        else:
            # 如果没有外部 label 顺序，则从 OBO 中生成全量 term 顺序。
            # 这种情况下，训练时必须使用这里保存的 go_terms_xx.txt 作为 logits 列顺序。
            terms = get_all_namespace_canonical_terms(
                ont=ont,
                ont_type=ont_type,
                include_root=include_root,
            )

        edges, output_terms, stats = build_go_edges(
            ont=ont,
            label_terms=terms,
            ont_type=ont_type,
            transitive=transitive,
        )
        
        edge_path = out_dir / f"go_edges_{ont_type}.pt"
        term_path = out_dir / f"go_terms_{ont_type}.txt"
        
        torch.save(edges.cpu().long(), edge_path)
        write_go_terms(term_path, output_terms)

        stats["edge_path"] = str(edge_path)
        stats["term_path"] = str(term_path)

        stats_by_type[ont_type] = stats

    return stats_by_type