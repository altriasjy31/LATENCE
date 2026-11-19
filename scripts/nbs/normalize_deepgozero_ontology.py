#!/usr/bin/env python3
"""Normalize frozen OBO/OWL using supplied DeepGOZero JARs with safe auxiliary IDs.

Default: corrected Java 11+ source-file backend (no Groovy needed).
--mode native: upstream Groovy diagnostic only; never emits a training go.norm.
The original uploaded source and JAR files are left unchanged.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dependency_free_source(text):
    """Strip only the known leading dependency annotation, preserve algorithm."""
    stripped = text.lstrip()
    if stripped.startswith('@Grapes(['):
        end = stripped.find('])')
        if end < 0:
            raise ValueError('Unterminated @Grapes annotation in Normalizer.groovy')
        stripped = stripped[end + 2:].lstrip()
    if re.search(r'@(?:Grab|Grapes|GrabConfig)\b', stripped):
        raise ValueError('Unexpected dependency annotations outside leading @Grapes block')
    expected = ['manager.loadOntologyFromOntologyDocument(ontologyFile)',
                'normalizer.normalize(iOntology, factory)',
                'renderer.render(rTranslator.visit(ax))']
    if any(token not in stripped for token in expected):
        raise ValueError('Unsupported Normalizer.groovy variant; inspect the supplied source')
    return stripped


def input_inventory(path):
    text = Path(path).read_text(encoding='utf-8')
    obo = bool(re.search(r'^format-version:\s', text, re.M))
    # Block import declarations, not ordinary linked database cross-references.
    imports = (re.findall(r'^import:\s*(\S+)', text, re.M) if obo else
               re.findall(r'(?:owl:imports\b[^>]*|Import\s*\([^)]*\)|<Import\b[^>]*>[^<]*)', text))
    if imports:
        raise ValueError('Input declares ontology imports. First resolve/freeze its complete import closure '
                         'locally and remove import declarations from the merged document; '
                         'the wrapper does not fetch changing external ontology content.')
    versions = re.findall(r'^(?:data-version|date):\s*(.+)', text, re.M) if obo else []
    return {'format_hint': 'OBO' if obo else 'OWL_or_other_OWLAPI_format',
            'declared_version_lines': versions,
            'obo_term_stanzas': len(re.findall(r'^\[Term\]\s*$', text, re.M)) if obo else None,
            'obo_intersection_of_lines': len(re.findall(r'^intersection_of:', text, re.M)) if obo else None,
            'obo_relationship_lines': len(re.findall(r'^relationship:', text, re.M)) if obo else None,
            'import_declarations': []}


def jar_inventory(directory):
    jars = sorted(Path(directory).glob('*.jar'))
    if not jars:
        raise FileNotFoundError(f'No JARs in {directory}')
    required = {
        'OWLManager': 'org/semanticweb/owlapi/apibinding/OWLManager.class',
        'JCEL': 'de/tudresden/inf/lat/jcel/owlapi/main/JcelReasoner.class',
        'OBO_parser': 'org/obolibrary/oboformat/parser/OBOFormatParser.class',
    }
    providers = {key: [] for key in required}
    records = []
    for path in jars:
        with zipfile.ZipFile(path) as jar:
            names = set(jar.namelist())
        for label, clazz in required.items():
            if clazz in names:
                providers[label].append(path.name)
        records.append({'name': path.name, 'sha256': sha256(path), 'size_bytes': path.stat().st_size})
    for label in ('OWLManager', 'JCEL'):
        if len(providers[label]) != 1:
            raise ValueError(f'Expected exactly one bundled provider for {label}, found {providers[label]}')
    return jars, records, providers


def parse_normal_forms(repo, path):
    """Use the exact original parser without importing torch/sklearn/click."""
    source = Path(repo) / 'deepgozero.py'
    tree = ast.parse(source.read_text())
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_normal_forms']
    if len(defs) != 1:
        raise ValueError('Original source lacks exactly one load_normal_forms definition')
    scope = {}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(source), 'exec'), scope)
    forms = scope['load_normal_forms'](str(path), {})
    nf1, nf2, nf3, nf4, relations, classes = forms
    nfs = [len(nf1), len(nf2), len(nf3), len(nf4)]
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not sum(nfs):
        raise ValueError('Generated go.norm has no subclass normal forms recognized by original DeepGOZero')
    subclass_lines = sum('SubClassOf' in line for line in lines)
    if sum(nfs) != subclass_lines:
        raise ValueError(f'{subclass_lines - sum(nfs)} SubClassOf lines ignored by the original parser; '
                         'inspect go.norm before using it for training')
    return {'normal_form_counts': dict(zip(('nf1', 'nf2', 'nf3', 'nf4'), nfs)),
            'parsed_classes': len(classes), 'parsed_GO_classes': sum(bool(re.fullmatch(r'GO:\d{7}', g)) for g in classes),
            'parsed_relations': len(relations), 'output_nonempty_lines': len(lines),
            'non_subclass_lines_not_consumed_by_model': len(lines) - subclass_lines}, classes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, required=True, help='Original deepgozero source with Normalizer.groovy and jars/')
    parser.add_argument('--ontology-file', type=Path, required=True, help='Frozen OBO/OWL without unresolved import declarations')
    parser.add_argument('--candidate-go-ids', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['corrected', 'native'], default='corrected')
    parser.add_argument('--java-bin', default='java', help='Java 11+ with source-file launcher / jdk.compiler')
    parser.add_argument('--groovy-bin', default='groovy')
    parser.add_argument('--jar-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Inspect source/input/JARs and print the command; no output directory created')
    args = parser.parse_args(argv)
    repo, ontology = args.repo_dir.resolve(), args.ontology_file.resolve()
    source = repo / 'Normalizer.groovy'
    run_source = dependency_free_source(source.read_text())
    inventory = input_inventory(ontology)
    jars, jar_records, providers = jar_inventory((args.jar_dir or repo / 'jars').resolve())
    if inventory['format_hint'] == 'OBO' and len(providers['OBO_parser']) != 1:
        raise ValueError('OBO input requires exactly one bundled OBO parser')
    candidate_ids = []
    if args.candidate_go_ids:
        candidate_ids = args.candidate_go_ids.read_text().splitlines()
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)) or any(
                re.fullmatch(r'GO:\d{7}', value) is None for value in candidate_ids):
            raise ValueError('Candidates must be unique canonical GO IDs, one per line')
    classpath = os.pathsep.join(str(p.resolve()) for p in jars)
    java_source = Path(__file__).with_name('DeepGOZeroNormalizer.java').resolve()
    executable = shutil.which(args.java_bin if args.mode == 'corrected' else args.groovy_bin)
    if args.mode == 'corrected':
        if not java_source.is_file():
            raise FileNotFoundError(f'Missing packaged Java backend: {java_source}')
        command = [executable or args.java_bin, '-cp', classpath, str(java_source), str(ontology), '<output-dir>']
    else:
        command = [executable or args.groovy_bin, '-cp', classpath,
                   '-Dgroovy.grape.enable=false', '<output-dir>/Normalizer.local.groovy',
                   '-i', str(ontology), '-o', '<output-dir>/native_diagnostic.norm']
    report = {'schema': 'original-deepgozero-normalization-v1', 'complete': False,
              'status': 'dry_run' if args.dry_run else 'pending',
              'source': {'path': str(source), 'sha256': sha256(source)},
              'run_source_sha256': hashlib.sha256(run_source.encode()).hexdigest(),
              'ontology': {'path': str(ontology), 'sha256': sha256(ontology), **inventory},
              'jars': jar_records, 'class_providers': providers,
              'mode': args.mode, 'backend': 'java-source-file' if args.mode == 'corrected' else 'native-groovy-diagnostic',
              'groovy_executable': shutil.which(args.groovy_bin), 'java_executable': shutil.which(args.java_bin),
              'command': command, 'dependency_policy': 'local JARs; no Grapes downloads',
              'corrected_backend_sha256': sha256(java_source) if args.mode == 'corrected' else None,
              'candidate_go_count': len(candidate_ids),
              'candidate_go_ids_sha256': sha256(args.candidate_go_ids) if args.candidate_go_ids else None,
              'normalizer_algorithm': ('JCEL shared translator factory plus typed NF1-NF4 renderer with collision-free auxiliary symbols'
                                       if args.mode == 'corrected' else 'original uploaded source; unsafe auxiliary factory behavior retained'),
              'known_limitations': [
                  'Original Groovy creates a separate JCEL factory; the corrected backend fixes that and maps fresh auxiliary class IDs.',
                  'Native Groovy output is diagnostic only because auxiliary IDs can collide even without an Ignoring message.',
                  'Model parser consumes only NF1-NF4 subclass axioms; property axioms are counted but not consumed.',
                  'OBO can be parsed; GO/GO-Plus OWL may contain different logical content even for the same date.',
                  'The supplied model consumes only class NF1-NF4. Other normalized axiom kinds are exported separately, not included in its loss.',
                  'The correction was checked on finite logical toy models; this is not an assertion of full GO-Plus semantic equivalence.']}
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if executable is None:
        raise FileNotFoundError('Java executable not found; install Java 11+ including jdk.compiler or pass --java-bin.'
                                if args.mode == 'corrected' else
                                'Groovy executable not found. Native diagnostic needs a compatible Groovy distribution with CliBuilder/commons-cli.')
    out = args.output_dir.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Choose a new normalization directory: {out}')
    out.mkdir(parents=True)
    if args.mode == 'corrected':
        command[-1] = str(out)
    else:
        # Keep invoked dependency-stripped script, never edit the source archive.
        local_script = out / 'Normalizer.local.groovy'
        local_script.write_text(run_source, encoding='utf-8')
        command[-5] = str(local_script)
        command[-1] = str(out / 'native_diagnostic.norm')
    report['command'] = command
    for binary, label in [(executable, 'engine'), (shutil.which(args.java_bin), 'java')]:
        if binary:
            check = subprocess.run([binary, '-version'], text=True, capture_output=True)
            report[label + '_version'] = (check.stdout + check.stderr).strip()
    with (out / 'normalizer.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    log = (out / 'normalizer.log').read_text()
    ignored = [line for line in log.splitlines() if line.startswith('Ignoring ')]
    report.update(returncode=result.returncode, ignored_renderer_axioms=len(ignored),
                  log_sha256=sha256(out / 'normalizer.log'))
    try:
        if result.returncode != 0:
            raise RuntimeError(f'{report["backend"]} normalizer failed with exit {result.returncode}; inspect {out / "normalizer.log"}')
        norm = out / ('go.norm' if args.mode == 'corrected' else 'native_diagnostic.norm')
        if not norm.is_file() or not norm.stat().st_size:
            raise RuntimeError('Normalizer produced no nonempty go.norm despite its exit status')
        parsed, classes = parse_normal_forms(repo, norm)
        report.update(parsed)
        report['go_norm_sha256'] = sha256(norm) if args.mode == 'corrected' else None
        report['native_diagnostic_sha256'] = sha256(norm) if args.mode == 'native' else None
        missing = [g for g in candidate_ids if g not in classes]
        report.update(candidate_go_supported=len(candidate_ids) - len(missing), candidate_go_missing=missing)
        if candidate_ids:
            (out / 'candidate_coverage.tsv').write_text('go_id\tin_parsed_normal_forms\n' +
                ''.join(f'{go}\t{go in classes}\n' for go in candidate_ids))
        if args.mode == 'corrected':
            import csv
            with (out / 'normalizer_stats.tsv').open() as stream:
                report['backend_counts'] = {row['key']: int(row['value']) for row in csv.DictReader(stream, delimiter='\t')}
            report['artifacts_sha256'] = {name: sha256(out / name) for name in
                ['go.norm', 'symbol_mapping.tsv', 'excluded_normal_forms.tsv', 'normalizer_stats.tsv',
                 'untranslated_input_axioms.tsv']}
            with (out / 'symbol_mapping.tsv').open() as stream:
                special = [row for row in csv.DictReader(stream, delimiter='\t') if row['special_builtin'] != 'none']
            report['special_builtin_classes'] = special
            report['special_builtin_semantics'] = 'The neural model does not automatically enforce universal/empty interpretations of these class symbols.'
            report.update(complete=True, status='normalized')
        else:
            report.update(complete=False, status='native_diagnostic_unsafe')
    except Exception as error:
        report.update(status='failed', error=str(error))
        (out / 'normalization_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise
    (out / 'normalization_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'normal_form_counts', 'candidate_go_supported',
                                           'candidate_go_missing', 'ignored_renderer_axioms')}, indent=2))
    print(out / ('go.norm' if args.mode == 'corrected' else 'native_diagnostic.norm'))


if __name__ == '__main__':
    main()
