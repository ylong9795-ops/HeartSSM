"""Partition the existing 1K cohort by case; never modify samples or labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'synthesis_data/anatomical_per_sample_production1000_v1/manifest.json'
OUTPUT = ROOT/'synthesis_data/anatomical_per_sample_production1000_neural_splits_v1'


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def partition(rows, counts, seed):
    """Keep all frames of one coordinate identity in one reproducible split."""
    if len(rows) != sum(counts) or min(counts) < 1:
        raise ValueError('Positive split sizes must sum to the complete source cohort')
    for field in ('candidate_id', 'path', 'sha256', 'coordinate_sha256'):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError('Duplicate source cases: '+field)
    ordered = sorted(rows, key=lambda row: (
        sha256(f"{seed}:{row['coordinate_sha256']}".encode()), row['coordinate_sha256']))
    a, b, _ = counts
    return dict(train=ordered[:a], validation=ordered[a:a+b], test=ordered[a+b:])


def prepare(source=SOURCE, output=OUTPUT, counts=(800, 100, 100), seed=2026, *, progress=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source.parent:
        raise ValueError('Split output must be separate from original source data')
    raw = source.read_bytes()
    manifest = json.loads(raw)
    if (not manifest.get('complete') or manifest.get('retired') or
            manifest.get('format') not in ('wholeheart-simple-training-v1', 'wholeheart-simple-training-v2')):
        raise ValueError('Complete, active simple source manifest required')
    plan = (source.parent/'plan.json').read_bytes()
    if sha256(plan) != manifest['plan_sha256']:
        raise ValueError('Original generation plan digest mismatch')
    if manifest['accepted_count'] != len(manifest['samples']):
        raise ValueError('Source cohort count mismatch')
    rows = []
    members = manifest['samples']
    if progress:
        from tqdm.auto import tqdm
        members = tqdm(members,desc='Checking 1K source files',unit='case',mininterval=1.)
    for original in members:
        row = dict(original)
        for field in ('path', 'qc_path'):
            path = (source.parent/row[field]).resolve()
            if not path.is_file():
                raise ValueError('Missing original source file: '+str(path))
            row[field] = os.path.relpath(path, output)
        rows.append(row)
    splits = partition(rows, counts, seed)
    provenance = dict(source_manifest=os.path.relpath(source, output),
        source_manifest_sha256=sha256(raw), source_plan_sha256=sha256(plan),
        source_samples=len(rows), split_seed=seed,
        algorithm='SHA256(seed:coordinate_sha256) ordering; train, validation, test',
        split_unit='complete case with all 24 independent phases and three views',
        preserved_source_files=True, preserved_sample_fields='all except relocated path and qc_path',
        counts={key:len(value) for key,value in splits.items()},
        test_policy='Diagnostic holdout only, evaluated after training; historical pilot may have used these cases.')
    payloads = {'plan.json':plan}
    for name, members in splits.items():
        result = dict(manifest, samples=members, split=name,
            accepted_count=len(members), requested_samples=len(members), target_accepted=len(members),
            source_generation_counts={key:manifest[key] for key in
                ('attempted', 'requested_samples', 'target_accepted') if key in manifest},
            derivation=provenance)
        payloads[name+'.json'] = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
    provenance['split_manifest_sha256'] = {name:sha256(payloads[name+'.json']) for name in splits}
    payloads['derivation.json'] = (json.dumps(provenance, indent=2, sort_keys=True)+'\n').encode()
    # Re-running is idempotent; never silently change a split used by a checkpoint.
    for name, content in payloads.items():
        path = output/name
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Existing split differs; choose a new output directory: '+str(path))
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = output/name
        if not path.exists():
            path.write_bytes(content)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print('Preparing/verifying existing 800/100/100 split; original labels remain unchanged.',flush=True)
    result = prepare(args.source, args.output, seed=args.seed, progress=True)
    print(json.dumps(dict(output=str(args.output), counts=result['counts'],
                          source_manifest_sha256=result['source_manifest_sha256']), indent=2),flush=True)


if __name__ == '__main__':
    main()
