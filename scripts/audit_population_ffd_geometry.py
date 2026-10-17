"""Full split endpoint geometry; no optimization or patient fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import default_collate
import yaml

from train_neural_ffd import data_identity, dataset_from_config, repository_path, to_device
from train_joint_refiner import audit_geometry, initialize_geometry
from utils.joint_refiner_training import load_refiner_checkpoint, sampled_composed_jacobians


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('val','test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1: parser.error('Positive geometry worker count required')
    torch.set_num_threads(2)
    model, saved = load_refiner_checkpoint(args.checkpoint, args.device)
    model.eval()
    data = yaml.safe_load(repository_path(saved['base_checkpoint']['config']['data_config']).read_text())
    if data_identity(data) != saved['base_checkpoint']['data_identity']:
        raise ValueError('Checkpoint dataset identity changed')
    dataset = dataset_from_config(data, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    records, pending = [], []
    def finish():
        future, jacobian = pending.pop(0)
        record = dict(future.result(), sampled_composed_jacobian=jacobian)
        records.append(record)
        (args.output/'cases.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
        print(f"Geometry {len(records)}/{len(dataset)} Base={record['base']['passed']} "
              f"Refiner={record['refiner']['passed']}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_geometry, initargs=(str(args.checkpoint.resolve()),)) as pool:
        for index in range(len(dataset)):
            batch = to_device(default_collate([dataset[index]]), args.device)
            output = model(batch, pose=batch['pose_gt'])
            jacobian = sampled_composed_jacobians(model.base_model, output['base_output'],
                                                 output['refinement']['controls_res_mm'])
            item = dict(case_id=batch['case_id'][0], base=output['vertices_base_4d'][0].cpu().numpy(),
                        refiner=output['vertices_4d'][0].cpu().numpy())
            pending.append((pool.submit(audit_geometry, item), jacobian))
            del output, batch, item
            if len(pending) >= args.workers*2:
                finish()
        while pending:
            finish()
    summary = dict(split=args.split, cases=len(records), checkpoint=str(args.checkpoint),
        geometry_base_pass=sum(r['base']['passed'] for r in records),
        geometry_refiner_pass=sum(r['refiner']['passed'] for r in records),
        geometry_regression_case_ids=[r['case_id'] for r in records if r['base']['passed'] and not r['refiner']['passed']],
        sampled_jacobians_positive=all(r['sampled_composed_jacobian']['refiner']['finite'] and
            r['sampled_composed_jacobian']['refiner']['nonpositive']==0 for r in records),
        scope='full split, actual24 surfaces plus all linear vertex chords including23->0; sampled local Jacobians, no whole-field certificate')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
