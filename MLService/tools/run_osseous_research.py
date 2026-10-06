"""Config CLI for the isolated author-ROI osseous baseline."""
import argparse
import json
import sys

from training.tmj_osseous_research import ResearchError, load_config, preflight, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--train', action='store_true')
    action.add_argument('--develop', action='store_true', help='Train/validation only; requires frozen split')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = preflight(config) if args.preflight else train(config, development=args.develop)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except ResearchError as error:
        print(json.dumps({'status': 'rejected', 'reason': str(error)}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({'status': 'failed', 'reason': 'research_execution_failed'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
