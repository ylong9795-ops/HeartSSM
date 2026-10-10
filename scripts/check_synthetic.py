"""Verify the data interface, make representative figures, optionally scan I/O.

Run from any cwd: python scripts/check_synthetic.py --run-tests --benchmark
The large scan validates native reading, never rerenders/generates any case.
The interface has no dependency on the EchoGSF checkout. Optional historic
evidence is labelled separately from fresh checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data.mask_transform import CHAMBERS, VIEWS, MaskTransform, _validity
from data.synthetic_4d import Synthetic4DDataset, _sha256


def _path(value):
    return str(ROOT / value)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if tuple(config['data']['views']) != VIEWS or tuple(config['data']['chambers']) != CHAMBERS:
        raise ValueError('Config must use canonical view/chamber order')
    if config['data']['phase_semantics'] != 'relative_observed_cycle_time':
        raise ValueError('Only observed relative cycle time is supplied to the encoder')
    return config


def dataset_from_config(config, split='train', *, native=False, topology=True):
    data = config['data']
    transform = MaskTransform(use_sdf=False) if native else MaskTransform(**config['mask'])
    return Synthetic4DDataset(_path(data[split + '_manifest']), _path(data['asset_path']), transform,
        verify=config['check']['verify_source_hash'],
        path_map={old: _path(new) for old, new in data['path_map'].items()},
        expected_phases=data['expected_phases'], remove_duplicate_endpoint=data['remove_duplicate_endpoint'],
        template=_path(data['template_path']) if topology else None,
        template_file_sha256=data['template_file_sha256'] if topology else None)


def _save(report, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'dataset_check.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def _record(report, name, status, **details):
    report['checks'].append(dict(name=name, status=status, **details))


def _volumes(sample, geometry):
    # Adapted from bundled outputs/synthetic_audit_20261009/audit.py.
    volumes = {}
    physical = sample['vertices_gt'].numpy()
    for chamber, (left, right) in zip(CHAMBERS, geometry['chamber_ranges'].tolist()):
        vertices = physical[:, left:right]
        tri = vertices[:, geometry['faces'][chamber].numpy()]
        cross = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
        volumes[chamber] = np.abs(np.einsum('tfi,tfi->tf',
            tri[:, :, 0] - vertices.mean(1)[:, None], cross).sum(1) / 6000)
    return volumes


def _rgb(masks):
    # Adapted from the bundled audit montage; overlap is retained in colour.
    colors = ['#ef5350', '#ffb74d', '#42a5f5', '#66bb6a']
    image = np.full((*masks.shape[-2:], 3), .04)
    for channel, color in enumerate(colors):
        inside = masks[channel] > 0
        image[inside] = np.maximum(image[inside], np.array(to_rgb(color)) * .9)
    return image


def _figure(sample, volumes, path):
    reference, es = 0, int(np.argmin(volumes['LV']))
    fig, axes = plt.subplots(6, 6, figsize=(14, 13))
    for view, view_name in enumerate(VIEWS):
        for column, frame in enumerate((reference, es)):
            axes[2 * view, 3 * column].imshow(_rgb(sample['native_masks'][frame, view].numpy()))
            axes[2 * view, 3 * column].set_title(f'{view_name} native f{frame}')
            axes[2 * view, 3 * column + 1].imshow(_rgb(sample['masks'][frame, view].numpy()))
            axes[2 * view, 3 * column + 1].set_title(f'{view_name} fixed FOV f{frame}')
            axes[2 * view, 3 * column + 2].plot(sample['phases'], volumes['LV'])
            axes[2 * view, 3 * column + 2].axvline(float(sample['phases'][frame]), color='r')
            axes[2 * view, 3 * column + 2].set_title('LV volume (mL)', fontsize=9)
        for chamber, chamber_name in enumerate(CHAMBERS):
            axis = axes[2 * view + 1, chamber]
            if 'sdf' in sample:
                sdf = sample['sdf'][reference, view, chamber].numpy()
                limit = max(float(np.abs(sdf).max()), 1.)
                axis.imshow(sdf, cmap='coolwarm', vmin=-limit, vmax=limit)
            else:
                axis.imshow(sample['masks'][reference, view, chamber], cmap='gray')
            for frame, color in ((reference, 'lime'), (es, 'yellow')):
                mask = sample['masks'][frame, view, chamber].numpy()
                if mask.any() and not mask.all():
                    axis.contour(mask, levels=[.5], colors=[color], linewidths=.6)
            axis.set_title(f'{view_name} {chamber_name} SDF', fontsize=9)
        for axis in axes[2 * view + 1, 4:]:
            axis.axis('off')
        for column in (0, 1, 3, 4):
            axes[2 * view, column].axis('off')
        for axis in axes[2 * view + 1, :4]:
            axis.axis('off')
    fig.suptitle(f"Case {sample['candidate_id']} | LV red, LA orange, RV blue, RA green\n"
                 'f0: source reference, green contour; LV Vmin: yellow contour. SDF blue inside, red outside.')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _check_case(dataset, index, output):
    sample = dataset[index]
    time_count = dataset.expected_phases
    for key in ('masks', 'native_masks'):
        if sample[key].shape[:3] != (time_count, 3, 4) or not torch.all((sample[key] == 0) | (sample[key] == 1)):
            raise ValueError('Invalid mask shape/binary: ' + key)
    if sample['vertices_gt'].shape != (time_count, dataset.vertex_count, 3):
        raise ValueError('Invalid GT shape')
    valid = _validity(time_count, sample['view_present'], sample['annotated'])
    if np.any(sample['masks'].numpy()[~valid]):
        raise ValueError('Unannotated masks are not zero placeholders')
    if 'sdf' in sample:
        sdf, masks = sample['sdf'].numpy(), sample['masks'].numpy().astype(bool)
        if sdf.shape != masks.shape or not np.isfinite(sdf).all() or np.any(sdf[~valid]):
            raise ValueError('Invalid SDF shape/finite/availability')
        active = np.broadcast_to(valid[..., None, None], masks.shape)
        if np.any(sdf[active & masks] >= 0) or np.any(sdf[active & ~masks] <= 0):
            raise ValueError('SDF sign mismatch')
    source = dataset.sources[index]
    with np.load(source['path'], allow_pickle=False) as archive:
        np.testing.assert_array_equal(sample['vertices_gt'], archive['vertices'][:-1].astype(float) * dataset.norm_to_mm)
        for key in ('probe_R', 'probe_t', 'probe_s'):
            np.testing.assert_array_equal(sample['pose_gt'][key], archive[key][:-1])
    if sample['clipped_foreground'].any():
        raise ValueError('Normalized foreground clipping')
    inverse = sample['model_to_native'].numpy()
    if np.any(np.linalg.det(inverse) <= 0) or np.any(inverse[..., [0, 1], [0, 1]] <= 0):
        raise ValueError('Mask transform flips orientation')
    native_area = sample['native_masks'].numpy().sum(axis=(-1, -2)) * sample['native_pixel_spacing'].numpy().prod(-1)[..., None]
    model_area = sample['masks'].numpy().sum(axis=(-1, -2)) * sample['pixel_spacing'].numpy().prod(-1)[..., None]
    volumes = _volumes(sample, dataset.shared_geometry)
    path = output / 'examples' / f"case_{sample['candidate_id']:08d}.png"
    _figure(sample, volumes, path)
    return dict(status='PASS', case_id=sample['case_id'], candidate_id=sample['candidate_id'],
        index=index, sample_sha256=sample['sample_sha256'], coordinate_sha256=sample['coordinate_sha256'],
        shapes={key: list(value.shape) for key, value in sample.items() if isinstance(value, torch.Tensor)},
        figure=str(path.relative_to(output)), lv_vmin_frame=int(np.argmin(volumes['LV'])),
        volumes_ml={key: value.tolist() for key, value in volumes.items()},
        native_area_mm2=native_area.tolist(), model_area_mm2=model_area.tolist(),
        max_relative_area_change=float(np.max(np.abs(model_area - native_area) / np.maximum(native_area, 1))),
        clipped_foreground_pixels=int(sample['clipped_foreground'].sum()),
        model_spacing_mm=sample['pixel_spacing'][0].tolist())


def _benchmarks(config, output, report):
    """Bounded 4-thread reads; only tiny results leave each worker."""
    datasets = [(split, dataset_from_config(config, split, native=True, topology=False))
                for split in ('train', 'val', 'test')]
    membership = [(split, data, index) for split, data in datasets for index in range(len(data))]
    sizes = config['check']['benchmark_sizes']
    report['benchmarks'] = []
    start = time.perf_counter()
    seen = set()
    failures = []
    byte_count = 0

    def read(entry):
        split, dataset, index = entry
        try:
            sample = dataset[index]
            return dict(split=split, index=index, case_id=sample['case_id'],
                        bytes=Path(dataset.sources[index]['path']).stat().st_size)
        except Exception as error:
            return dict(split=split, index=index, error=str(error))

    with ThreadPoolExecutor(max_workers=4) as executor:
        for first in range(0, min(max(sizes), len(membership)), 4):
            for completed, result in enumerate(executor.map(read, membership[first:first + 4]), first + 1):
                if 'error' in result:
                    failures.append(result)
                elif result['case_id'] in seen:
                    failures.append(dict(result, error='Duplicate case identity'))
                else:
                    seen.add(result['case_id'])
                    byte_count += result['bytes']
                if completed in sizes:
                    elapsed = time.perf_counter() - start
                    milestone = dict(cases=completed, unique_cases=len(seen), status='FAIL' if failures else 'PASS',
                        seconds=elapsed, cases_per_second=completed / elapsed, source_bytes=byte_count,
                        compressed_MB_per_second=byte_count / 1e6 / elapsed, failures=list(failures),
                        scope='Native masks + saved GT + pose + hashes/QC/closure, no SDF/warp/render; TEST data integrity only')
                    report['benchmarks'].append(milestone)
                    _save(report, output)
                    print(f"Native read {completed}: {milestone['status']}, {milestone['cases_per_second']:.2f} cases/s", flush=True)
            if (first // 4) % 125 == 0:
                print(f'Native read progress: {min(first + 4, len(membership))}', flush=True)
    if max(sizes) > len(membership):
        _record(report, 'full_cohort_read', 'FAIL', reason='Requested scan exceeds source membership')
    else:
        _record(report, 'full_cohort_read', 'FAIL' if failures else 'PASS', cases=min(max(sizes), len(membership)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/synthetic_data.yaml')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    output = Path(_path(config['check']['output']))
    (output / 'examples').mkdir(parents=True, exist_ok=True)
    report = dict(status='UNVERIFIED', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  config=config, config_sha256=_sha256(args.config), checks=[], cases=[],
                  versions={name: importlib.metadata.version(name) for name in
                            ('numpy', 'torch', 'scipy', 'opencv-python-headless', 'PyYAML', 'matplotlib')})
    provenance_paths = ['data_generation/dataset_io.py', 'data_generation/utils.py',
        'src/heart4d/shared/data/source_data.py', 'src/heart4d/shared/data/synthetic.py',
        'src/heart4d/shared/data/ed.py', 'src/heart4d/s1_shape/dataset.py',
        'src/heart4d/s1_shape/observation.py', 'src/heart4d/s1_shape/contracts.py',
        'src/heart4d/s1_shape/checkpoint.py', 'tests/s1/unit/test_dataset.py',
        'configs/data/synthetic_dev_1k.yaml', 'outputs/synthetic_audit_20261009/audit.py',
        'outputs/synthetic_audit_20261009/visualize.py']
    bundle = ROOT / 'synthetic_bundle_20261009'
    if (bundle / 'BUNDLE_INDEX.json').exists():
        index = json.loads((bundle / 'BUNDLE_INDEX.json').read_text())
        expected = {row['path']: row['sha256'] for row in index['files']}
        report['source_code'] = dict(repository='https://github.com/ylong9795-ops/echo_gsf_4ch',
            branch='research/shape-p0-p1-20261008', commit=index['git_commit'],
            files={path: dict(sha256=_sha256(bundle / path), matches_bundle=_sha256(bundle / path) == expected[path])
                   for path in provenance_paths})
        _record(report, 'reused_source_versions', 'PASS' if all(
            row['matches_bundle'] for row in report['source_code']['files'].values()) else 'FAIL')
    else:
        _record(report, 'reused_source_versions', 'UNVERIFIED', reason='Bundle index unavailable')
    report['implementation_sha256'] = {path: _sha256(ROOT / path) for path in
        ('data/synthetic_4d.py', 'data/mask_transform.py', 'scripts/check_synthetic.py', 'tests/test_synthetic_4d.py')}
    report['reuse_decisions'] = dict(
        ported=['Streaming file SHA256 and float32 coordinate hash', 'Visibility policy and pose/mm/closure checks',
                'Template topology digest', 'Audit volume formula and overlap-preserving mask colours'],
        adapted=['Manifest paths via explicit relocation map; no source writes',
                 'Lazy source cases into standard collatable PyTorch dictionaries',
                 'Physical inverse nearest-neighbour warp with time, anisotropic spacing, configurable size and affine output',
                 'Identity/tamper and reference-vs-cycle unit tests'],
        added=['Finite empty/full SDF, clipping and optional normalization',
               'Boundary extraction, per-frame validity, clipping rejection',
               'Integrated check report, visual comparisons and bounded throughput checks'],
        excluded=['ED-only repaired manifests are rejected for full-cycle reading',
                  'ED NPY first-frame optimization is unnecessary for correctness and not ported'])
    try:
        data = dataset_from_config(config, topology=config['check']['verify_topology'])
        if data.shared_geometry is None:
            raise ValueError('Representative geometry checks require the fixed shared template')
        report['dataset'] = dict(cases=len(data), manifest_sha256=data.manifest_sha256,
            asset_sha256=data.asset_sha256, template_sha256=data.template_sha256,
            template_artifact_sha256=data.shared_geometry['artifact_sha256'],
            coordinate_system='centered physical mm = saved normalized vertices * scale_mm',
            center_mm=data.asset['center_mm'], norm_to_mm=data.norm_to_mm,
            view_order=list(VIEWS), chamber_order=list(CHAMBERS),
            phase_semantics='Relative observed cycle time; phases_gt is supervision only; no known ED input required')
        _record(report, 'source_index_asset_plan_topology', 'PASS')
        indexes = np.linspace(0, len(data) - 1, min(config['check']['sample_count'], len(data)), dtype=int)
        for index in indexes:
            try:
                result = _check_case(data, int(index), output)
            except Exception as error:
                result = dict(status='FAIL', index=int(index), error=str(error))
            report['cases'].append(result)
            print(f"Representative index {index}: {result['status']}", flush=True)
            _save(report, output)
        _record(report, 'representative_masks_sdf_gt_pose',
                'PASS' if all(row['status'] == 'PASS' for row in report['cases']) else 'FAIL', count=len(indexes))
        # Pick small, median and large *observed* LV areas for a compact montage.
        passing = [row for row in report['cases'] if row['status'] == 'PASS']
        selected = sorted(passing, key=lambda row: row['native_area_mm2'][0][0][0])
        if selected:
            montage, axes = plt.subplots(3, 6, figsize=(12, 7))
            for row, result in enumerate([selected[0], selected[len(selected) // 2], selected[-1]]):
                sample = data[result['index']]
                for column, frame in enumerate((0, result['lv_vmin_frame'])):
                    for view in range(3):
                        axes[row, column * 3 + view].imshow(_rgb(sample['masks'][frame, view].numpy()))
                        axes[row, column * 3 + view].set_title(f"ID {sample['candidate_id']} {VIEWS[view]} f{frame}", fontsize=8)
                        axes[row, column * 3 + view].axis('off')
            montage.suptitle('Small / median / large observed LV area | fixed FOV | f0 and measured LV Vmin')
            montage.tight_layout()
            montage.savefig(output / 'examples/representative_cases.png', dpi=140)
            plt.close(montage)
        try:
            for batch_size, workers in ((1, 0), (2, 0), (2, config['check']['batch_workers'])):
                loader = DataLoader(Subset(data, indexes[:2].tolist()), batch_size=batch_size,
                                    num_workers=workers, prefetch_factor=1 if workers else None,
                                    timeout=30 if workers else 0)
                batch = next(iter(loader))
                if batch['masks'].shape[0] != batch_size or len(set(batch['case_id'])) != batch_size:
                    raise ValueError('Batch shape/identity mismatch')
                for item, index in enumerate(indexes[:batch_size]):
                    expected = data[int(index)]
                    for key in ('masks', 'vertices_gt', 'sdf'):
                        if key in expected and not torch.equal(batch[key][item], expected[key]):
                            raise ValueError('Batch case alignment mismatch: ' + key)
                print(f'Batch B={batch_size}, workers={workers}: PASS', flush=True)
            _record(report, 'dataloader', 'PASS', batch_sizes=[1, 2], workers=[0, config['check']['batch_workers']])
        except Exception as error:
            _record(report, 'dataloader', 'FAIL', error=str(error))
    except Exception as error:
        _record(report, 'source_interface', 'FAIL', error=str(error))
    if args.run_tests:
        run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                             cwd=ROOT, capture_output=True, text=True)
        (output / 'unit_tests.log').write_text(run.stdout + run.stderr)
        count = re.search(r'Ran (\d+) tests', run.stderr)
        _record(report, 'unit_tests', 'PASS' if run.returncode == 0 else 'FAIL',
                count=int(count.group(1)) if count else 0, log='unit_tests.log')
    else:
        _record(report, 'unit_tests', 'UNVERIFIED', reason='Run --run-tests to obtain fresh evidence')
    _record(report, 'orientation_apex_chamber_anatomy', 'UNVERIFIED',
            reason='Positive affine/no flips checked; anatomical correctness requires review of figures or independent landmarks')
    spot_path = output / 'slice_spot_check.json'
    if spot_path.exists() and 'dataset' in report:
        spot = json.loads(spot_path.read_text())
        matched = (spot['dataset_manifest_sha256'] == report['dataset']['manifest_sha256'] and
                   _sha256(ROOT / spot['renderer_source']) == spot['renderer_sha256'] and
                   all(data.sources[row['index']]['sha256'] == row['sample_sha256'] for row in spot['cases']))
        _record(report, 'fresh_mesh_to_mask_oracle', spot['status'] if matched else 'UNVERIFIED',
                evidence='slice_spot_check.json', scope=spot['scope'],
                reason='Recorded spot evidence must match manifest, case hashes and renderer source')
    else:
        _record(report, 'fresh_mesh_to_mask_oracle', 'UNVERIFIED',
                reason='Dataset correspondence/hash checks do not independently rerender GT slices')
    _record(report, 'cross_chamber_real_population_certification', 'UNVERIFIED',
            reason='Existing bundle audit records limited A3C aorta coverage, inherited correspondence, and no full cross-chamber certification')
    report['source_limitations'] = ['24 output phases are linearly resampled from 10 source phases',
        'Frame zero is a source reference, not every chamber volume maximum',
        'A3C usually lacks aorta coverage; use as a third left-heart slice within this source scope',
        'Inter-chamber exclusion, real population coverage and material-point correspondence remain unverified',
        'Pilot candidate 939 has an ED-only 17-pixel A3C-LV correction; full-cycle data are never silently patched',
        'Fixed FOV inherited from a recorded pilot contract; this checker rejects foreground clipping',
        'Manifest accept describes the source QC scope, not new anatomical certification']
    if args.benchmark:
        try:
            _benchmarks(config, output, report)
        except Exception as error:
            _record(report, 'full_cohort_read', 'FAIL', error=str(error))
    else:
        _record(report, 'full_cohort_read', 'UNVERIFIED', reason='Run --benchmark for 100/1K/10K native I/O milestones')
    report['status'] = 'FAIL' if any(row['status'] == 'FAIL' for row in report['checks']) else 'PASS'
    report['status_scope'] = 'Implemented data-interface checks; UNVERIFIED items are not passed'
    report['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    _save(report, output)
    print(f"{report['status']}: {output / 'dataset_check.json'}", flush=True)
    return 1 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
