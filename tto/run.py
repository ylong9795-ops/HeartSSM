"""python -m tto.run --config configs/real_tto_v1.yaml --patients C01 C02 --stage all"""
import argparse

from .config import load_config
from .pipeline import RealTTOPipeline, STAGES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/real_tto_v1.yaml')
    parser.add_argument('--patients', nargs='+', required=True)
    parser.add_argument('--stage', choices=('all', 'preflight', *STAGES), default='all')
    parser.add_argument('--gpu', help='CUDA index, e.g.4; or cpu')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.gpu:config['runtime']['device'] = 'cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}'
    if args.batch_size is not None:
        if args.batch_size < 1:parser.error('--batch-size must be positive')
        config['runtime']['batch_size'] = args.batch_size
    if args.checkpoint:config['prior']['checkpoint'] = args.checkpoint
    if args.output_dir:config['output']['root'] = args.output_dir
    if args.data_root:config['data']['root'] = args.data_root
    result = RealTTOPipeline(config).run(args.patients, stage=args.stage, resume=args.resume)
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
