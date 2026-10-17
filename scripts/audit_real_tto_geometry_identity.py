"""Check original geometry-package identities saved by committed TTO exports.

This reads certificates and source bytes only. It does not run geometry kernels
or admit any reconstruction, and it never changes the live scientific sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'synthetic_bundle_20261009'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(value):
    if isinstance(value, dict):
        if 'source_implementation_sha256' in value and 'support_sha256' in value:
            yield value
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/real_tto_joint_v1_20261016')
    args = parser.parse_args()
    root = Path(args.output)
    if not root.is_absolute():
        root = ROOT / root
    protocol = read(root / 'protocol.json')
    config = protocol['scientific_config']
    locations = [(f'synthetic/SYN_{i:03d}', root / 'synthetic_hidden_gt/patients' / f'SYN_{i:03d}')
                 for i in range(config['synthetic']['cases'])]
    for arm in config['experiment']['arms']:
        parent = root if arm == 'joint' else root / 'ablations' / arm
        locations.extend((f'{arm}/{name}', parent / 'patients' / name)
                         for name in config['experiment']['cohort'])
    reference = None
    reference_export = None
    records, unavailable, errors = [], [], []
    for name, folder in locations:
        certificate = folder / 'export/geometry.json'
        if not all(p.exists() for p in (certificate, folder / 'export/completed.json', folder / 'states/export.pth')):
            unavailable.append(dict(export=name, failure_record_exists=(folder / 'failure.json').exists()))
            continue
        saved = list(identities(read(certificate)))
        if not saved:
            errors.append(dict(export=name, error='No saved original-geometry source identity'))
        if saved and reference is None:
            reference, reference_export = saved[0], name
        same = bool(saved) and all(item == reference for item in saved)
        if not same:
            errors.append(dict(export=name, error='Saved geometry identity differs from reference'))
        records.append(dict(export=name, certificate_sha256=sha(certificate),
                            saved_identity_records=len(saved), matches_reference=same))
    sources = []
    archive = root / 'independent_audit/geometry_source_snapshot'
    if reference is not None:
        paths = dict(reference['source_implementation_sha256'])
        paths['configs/geometry/source_support.json'] = reference['support_sha256']
        for relative, expected in paths.items():
            current = BUNDLE / relative
            actual = sha(current) if current.exists() else None
            matches = actual == expected
            if not matches:
                errors.append(dict(file=relative, error='Current original geometry source/support differs',
                                   expected_sha256=expected, actual_sha256=actual))
            target = archive / relative
            if matches:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha(target) != expected:
                    errors.append(dict(file=relative, error='Existing geometry source snapshot differs'))
                elif not target.exists():
                    target.write_bytes(current.read_bytes())
            sources.append(dict(file=relative, expected_sha256=expected, actual_sha256=actual,
                                current_matches=matches,
                                snapshot_matches=target.exists() and sha(target) == expected))
    result = dict(updated=time.time(), auditor_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(root / 'protocol.json'), reference_export=reference_export,
                  reference_identity=reference, checked_exports=len(records),
                  expected_export_locations=len(locations), unavailable=unavailable,
                  current_committed_sources_match=bool(records) and reference is not None and not errors,
                  errors=errors, sources=sources, exports=records,
                  scope='Saved original geometry identity entries and current source/support bytes only. '
                        'Source snapshots are archival copies, not a new kernel implementation. '
                        'No geometry recomputation, physics/GT accuracy certificate or full-cohort completion claim. '
                        'Failed/pending exports stay unavailable; package binaries and unlisted transitive modules are not hashed here.')
    path = root / 'independent_audit/geometry_backend_source_identity.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(dict(checked_exports=len(records), expected_export_locations=len(locations),
                          current_committed_sources_match=result['current_committed_sources_match'],
                          source_files=len(sources), errors=errors)))
    return 0 if result['current_committed_sources_match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
