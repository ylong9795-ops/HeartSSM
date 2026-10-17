"""Fixed-endpoint input ablation; no updates or validation-driven model selection."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

from data.mask_transform import CHAMBERS
from train_neural_ffd import data_identity, dataset_from_config, reconstruction_metrics, repository_path, to_device
from utils.joint_refiner_training import load_refiner_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, saved = load_refiner_checkpoint(args.checkpoint, args.device)
    model.eval()
    data = yaml.safe_load(repository_path(saved['base_checkpoint']['config']['data_config']).read_text())
    if data_identity(data) != saved['base_checkpoint']['data_identity']:
        raise ValueError('Ablation dataset identity changed')
    dataset = dataset_from_config(data, 'val')
    records = []
    for index in saved['experiment']['val_indices']:
        batch = to_device(default_collate([dataset[index]]), args.device)
        base = model.base_model(batch)
        features = model.observation_features(base, batch, pose=batch['pose_gt'])['features']
        modes = dict(base=None, full_observation=features)
        for name, scale in (('support_only', 0.), ('reversed_vectors', -1.)):
            changed = features.clone()
            for chamber in range(4):
                changed[:, :, chamber*4:chamber*4+3] *= scale
            modes[name] = changed
        modes['no_evidence'] = torch.zeros_like(features)
        row = dict(case_id=batch['case_id'][0], metrics={})
        for name, x in modes.items():
            prediction = base['vertices_4d'] if x is None else model.refine_base(
                base, dict(features=x), phases=batch['phases'])['vertices_4d']
            if name == 'no_evidence' and not torch.equal(prediction, base['vertices_4d']):
                raise ValueError('No-evidence inference must remain exactly neutral')
            row['metrics'][name] = reconstruction_metrics(prediction, batch['vertices_gt'],
                model.base_model.chamber_ranges, model.base_model.faces)
        records.append(row)
        print(f"Ablation {len(records)}/{len(saved['experiment']['val_indices'])}", flush=True)
    summary = {mode: {metric: float(np.sqrt(np.mean([r['metrics'][mode][c][metric]**2
        for r in records for c in CHAMBERS])))
        for metric in ('reference_vertex_rmse_mm', 'trajectory_vertex_rmse_mm', 'relative_motion_rmse_mm')}
        for mode in records[0]['metrics']}
    report = dict(scope='fixed VAL input ablation of an already frozen endpoint; no training/selection/TEST',
                  summary=summary, cases=records)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
