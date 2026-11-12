#!/usr/bin/env python3
"""Refresh GO closures in a Swiss-Prot pickle or parse a new DAT snapshot.

Requires pandas. The DAT reader comes from the supplied DeepGOZero script.
The required GO helpers are included from the supplied DeepGO2 source.
"""

import argparse
import gzip
import os
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

# GO propagation and evidence rules from the supplied deepgo2/deepgo/utils.py.
# BSD 3-Clause License
#
# Copyright (c) 2022, Bio-Ontology Research Group
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

class Ontology:
    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ancestors = {}


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

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        if term_id in self.ancestors:
            return self.ancestors[term_id]
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        self.ancestors[term_id] = term_set
        return term_set


def raw_frame(swissprot_file, keep_all, is_exp_code, is_cafa_target):
    proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros = load_data(swissprot_file)
    if not proteins or any(not protein for protein in proteins):
        raise ValueError(f"No valid Swiss-Prot entries found in {swissprot_file}")
    frame = pd.DataFrame({
        "proteins": proteins,
        "accessions": accessions,
        "genes": genes,
        "sequences": sequences,
        "annotations": annotations,
        "string_ids": string_ids,
        "orgs": orgs,
        "interpros": interpros,
    })
    frame["exp_annotations"] = [
        [go_id for go_id, code in (item.split("|", 1) for item in row) if is_exp_code(code)]
        for row in frame["annotations"]
    ]
    if not keep_all:
        frame = frame.loc[frame["exp_annotations"].map(bool)].copy()
    frame = frame.reset_index()
    frame["cafa_target"] = frame["orgs"].map(is_cafa_target)
    print(f"DAT entries: {len(proteins):,}; output rows: {len(frame):,}")
    return frame


def load_existing(path):
    frame = pd.read_pickle(path)
    required = {"proteins", "sequences", "annotations", "exp_annotations"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"Expected a Swiss-Prot DataFrame with {sorted(required)}. Use the DAT mode if needed.")
    return frame


def refresh_closures(frame, ontology):
    by_term = {}
    propagated = []
    changed = 0
    previous = frame.get("prop_annotations")
    for row, terms in enumerate(frame["exp_annotations"]):
        if isinstance(terms, (str, bytes)) or not hasattr(terms, "__iter__"):
            raise ValueError(f"Expected a collection of GO IDs in exp_annotations at row {row}")
        closure = set()
        for term in terms:
            if term not in by_term:
                by_term[term] = frozenset(ontology.get_ancestors(term))
            closure.update(by_term[term])
        if previous is not None:
            changed += set(previous.iloc[row]) != closure
        propagated.append(sorted(closure))
    frame["prop_annotations"] = propagated
    missing = sorted(term for term, ancestors in by_term.items() if not ancestors)
    print(f"Experimental GO IDs: {len(by_term):,}; no ancestor closure: {len(missing):,}")
    if missing:
        print("Examples without closure:", ", ".join(missing[:12]))
    if previous is not None:
        print(f"Rows with changed propagated annotations: {changed:,} / {len(frame):,}")


def save_new_pickle(frame, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{output.name}.", suffix=".pkl", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_pickle(temporary)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-pickle", type=Path, help="Refresh prop_annotations with the same rows and indices")
    source.add_argument("--swissprot-file", "-sf", type=Path, help="Parse the original UniProt DAT or DAT.GZ")
    parser.add_argument("--go-file", type=Path, required=True, help="Ontology OBO file for propagation")
    parser.add_argument("--out-file", "-o", type=Path, required=True, help="Path for a new pickle file")
    parser.add_argument("--keep-all", action="store_true", help="In DAT mode, retain entries without experimental evidence")
    args = parser.parse_args()
    if args.keep_all and args.input_pickle:
        parser.error("--keep-all is only used with --swissprot-file")
    if args.out_file.exists() or args.out_file.is_symlink():
        parser.error(f"Output already exists: {args.out_file}")

    print(f"GO file: {args.go_file.resolve()}")
    with args.go_file.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("["):
                break
            if line.startswith(("data-version:", "date:")):
                print(line.strip())
    ontology = Ontology(str(args.go_file), with_rels=True)
    if args.input_pickle:
        frame = load_existing(args.input_pickle)
    else:
        frame = raw_frame(args.swissprot_file, args.keep_all, is_exp_code, is_cafa_target)
    refresh_closures(frame, ontology)
    save_new_pickle(frame, args.out_file)
    print(f"Saved {len(frame):,} rows to {args.out_file.resolve()}")


def load_data(swissprot_file):
    """
    Parses UniProtKB data file and loads list of proteins and their
    annotations to lists
    Args:
       swissprot_file (string): A path to the data file
    Returns:
       Tuple of 8 lists (proteins, accessions, sequences, string_ids,
       orgs, genes, interpros)
    """

    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    string_ids = list()
    orgs = list()
    genes = list()
    interpros = list()
    opener = gzip.open if swissprot_file.suffix == '.gz' else open
    with opener(swissprot_file, 'rt', encoding='utf-8') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        strs = list()
        iprs = list()
        gene_id = ''
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    string_ids.append(strs)
                    orgs.append(org)
                    genes.append(gene_id)
                    interpros.append(iprs)
                prot_id = items[1]
                annots = list()
                strs = list()
                iprs = list()
                seq = ''
                gene_id = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                elif items[0] == 'STRING':
                    str_id = items[1]
                    strs.append(str_id)
                elif items[0] == 'GeneID':
                    gene_id = items[1]
                elif items[0] == 'InterPro':
                    ipr_id = items[1]
                    iprs.append(ipr_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        string_ids.append(strs)
        orgs.append(org)
        genes.append(gene_id)
        interpros.append(iprs)
    return proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros


if __name__ == "__main__":
    main()
