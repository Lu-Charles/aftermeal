"""Command-line tools for paired-photo food estimation."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Estimate remaining-food fraction from paired meal photos.')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('verify')
    command = sub.add_parser('benchmark')
    command.add_argument('--output', type=Path, default=Path('reports/my-benchmark.json'))
    args = parser.parse_args()
    if args.command == 'verify':
        from .benchmark import verify_payload
        print(f'Verified {verify_payload()} bundled files.')
    if args.command == 'benchmark':
        from .benchmark import run
        print(json.dumps(run(args.output)['summary'], indent=2))
if __name__ == '__main__':
    main()
