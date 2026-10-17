"""Single entry: python -m tto.run. This scaffold performs no optimization."""
import argparse
import json

from .config import load_config
from .experiments import CAPABILITIES
from .io import jsonable
from .pipeline import TTOPipeline
from .validate import validate_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/tto.yaml')
    parser.add_argument('--stage', choices=CAPABILITIES, default='preflight')
    parser.add_argument('--patients', nargs='+')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--state', help='New-schema state file, for --stage validate')
    parser.add_argument('--describe', action='store_true', help='Print stage implementation status and exit')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(CAPABILITIES, indent=2))
        return 0
    try:
        if args.stage == 'validate':
            if not args.state:
                raise ValueError('--stage validate requires --state')
            report = validate_state(args.state)
        else:
            if args.state:
                raise ValueError('--state is only valid with --stage validate')
            config = load_config(args.config)
            if args.data_root:
                config['data']['root'] = args.data_root
            if args.output_dir:
                config['output']['root'] = args.output_dir
            report = TTOPipeline(config).run(args.patients, stage=args.stage, resume=args.resume)
        print(json.dumps(jsonable(report), indent=2, ensure_ascii=False))
        return 1 if report.get('failures') else 0
    except (ValueError, NotImplementedError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
