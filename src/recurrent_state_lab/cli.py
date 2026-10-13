"""Run the fixed local recipe in a new output directory; preserve failures."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True, help='New result directory; existing paths are refused.')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--train-examples', type=int, default=1024)
    parser.add_argument('--seeds', default='7,19', help='One or two non-negative seeds, separated by a comma.')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists. Choose a new directory; existing data is preserved.')
    try:
        seeds = tuple(int(x) for x in args.seeds.split(','))
    except ValueError:
        parser.error('Seeds must be comma-separated integers.')
    from .experiment import Protocol, run, write_json
    from .report import render_report
    protocol = Protocol(seeds=seeds, epochs=args.epochs, train_examples=args.train_examples)
    report = None
    try:
        protocol.validate()
        report = run(args.output, protocol)
        with (args.output/'report.html').open('x', encoding='utf-8') as handle:
            handle.write(render_report(report))
    except (ValueError, AssertionError, RuntimeError, OSError) as error:
        if report is not None and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                'status': 'Incomplete; existing intermediate results retained.'})
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'status': 'complete', 'report': str(args.output/'report.html'),
                      'models': len(report['runs'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
