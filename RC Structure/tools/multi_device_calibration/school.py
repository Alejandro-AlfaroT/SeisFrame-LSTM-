"""V3-results intensity-calibration commands; preparation never launches NTHA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

RC = Path(__file__).resolve().parents[2]
PILOT = 'tools/multi_device_calibration'
DEFAULT_RUN = 'wcalV3_r1'
ROOTS = [f'outputs/v3DesignVerification100_V2_CD_Fc6_20261008_shard_{n:02d}'
         for n in range(1, 5)]
ROSTER = 'pilots/v6SystemSmoke10/screening_plan.json'          # the roster of the round being calibrated
RUNNER = 'Data_Generation/Run_Weighted_Seismic_Calibration.py'
PLOTTER = f'{PILOT}/plot_peak_hinges.py'


def call(arguments, allowed=(0,)):
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', *arguments], cwd=RC)
    if result.returncode not in allowed:
        raise RuntimeError(f'Command stopped with exit code {result.returncode}; preserve its reports and logs.')


def execute_with_plots(commands, plot_command):
    """Plot usable saved evidence even after a stop; preserve the original failure."""
    try:
        for command in commands:
            call(command)
    except (RuntimeError, KeyboardInterrupt):
        try:
            call(plot_command)
        except (RuntimeError, KeyboardInterrupt) as exc:
            print(f'Plot export also stopped: {exc}', file=sys.stderr)
        raise
    call(plot_command)


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        raise ValueError('Run name must contain letters, numbers, underscores or hyphens.')
    prepared = f'{PILOT}/prepared/{args.run_name}'
    # Inventory is frozen input too. Never rewrite it underneath an existing plan.
    if (RC / prepared / 'plan.json').exists():
        raise FileExistsError('This plan already exists; use it unchanged or choose a new --run-name.')
    call([f'{PILOT}/verify_setup.py'])
    roots = [part for root in (args.design_root or ROOTS) for part in ('--design-root', root)]
    mode = [] if args.require_qualified else ['--allow-open-m1-diagnostic']
    call(['Data_Generation/Inspect_Calibration_Designs.py', *roots,
          '--case-manifest', ROSTER, '--output', prepared, *mode], allowed=(0, 2))
    inventory = json.loads((RC / prepared / 'design_inventory.json').read_text(encoding='utf-8'))
    if args.action == 'inventory':
        return
    if inventory['excluded_count'] and not args.allow_excluded:
        raise ValueError('Excluded cases are listed in design_inventory.json. Review them; use --allow-excluded only for a deliberate eligible-subset diagnostic.')
    if inventory['eligible_count'] < args.devices:
        raise ValueError('Too few eligible cases for the requested devices; review the inventory.')
    call(['Data_Generation/Build_Weighted_Seismic_Calibration_Plan.py', *roots, *mode,
          '--case-manifest', f'{prepared}/eligible_cases.json',
          '--set-name', 'peer_mle_all', '--all-pairs-in-set', '--max-motion-points', '15000',
          '--assignment', 'one-per-case', '--seed', '20261006', '--shards', str(args.devices),
          '--workers', '1', '--minimum-scale', '1', '--initial-scale', '1', '--maximum-scale', '3.5',
          '--max-trials', '7', '--timeout-seconds', '3600',
          '--output-root', f'outputs/{args.run_name}', '--output', f'{prepared}/plan.json'])
    print(f'Copy {prepared} and all four COMPLETE design roots unchanged to every device. No analysis launched.')


def run(args):
    base = [RUNNER, '--plan', args.plan, '--shard', str(args.device)]
    if args.summarize_only:
        call([*base, '--summarize-only'])
        return
    call([f'{PILOT}/verify_setup.py'])
    call([*base, '--preflight-only'])
    if args.preflight_only:
        return
    # Recheck the saved first search on resume as well. A failed/unmet first
    # search must never be bypassed by using --resume.
    commands = [[*base, '--first-job-only', *(['--resume'] if args.resume else [])]]
    if not args.first_job_only:
        commands.append([*base, '--resume'])
    execute_with_plots(commands, [PLOTTER, '--plan', args.plan, '--device', str(args.device)])


def collect(args):
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = RC / plan_path
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    roots = args.device_root or [f"{plan['output_root']}/device_{n:02d}"
                                 for n in range(1, plan['shards'] + 1)]
    output = args.output or plan['output_root'] + '_collection'
    root_arguments = [part for root in roots for part in ('--device-root', root)]
    command = ['Data_Generation/Collect_Weighted_Seismic_Calibration.py', '--plan', args.plan,
               '--output', output, *root_arguments, *(['--allow-partial'] if args.allow_partial else [])]
    execute_with_plots([command], [PLOTTER, '--plan', args.plan, *root_arguments,
                                 '--output', str(Path(output)/'peak_beam_hinge_plots')])


def plots(args):
    command = [PLOTTER, '--plan', args.plan, '--member-scope', args.member_scope]
    if args.device:
        command += ['--device', str(args.device)]
    for root in args.device_root or []:
        command += ['--device-root', root]
    if args.output:
        command += ['--output', args.output]
    call(command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('setup', help='Check source, environment, launchers and records; no designs needed')
    for action in ('inventory', 'prepare'):
        p = sub.add_parser(action)
        p.add_argument('--design-root', action='append', help='Inside RC Structure; repeat for all four roots')
        p.add_argument('--run-name', default=DEFAULT_RUN)
        p.add_argument('--devices', type=int, choices=range(1, 5), default=4)
        p.add_argument('--allow-excluded', action='store_true')
        p.add_argument('--require-qualified', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--device', type=int, choices=range(1, 5), required=True)
    p.add_argument('--plan', default=f'{PILOT}/prepared/{DEFAULT_RUN}/plan.json')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--first-job-only', action='store_true')
    mode.add_argument('--summarize-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('collect')
    p.add_argument('--plan', default=f'{PILOT}/prepared/{DEFAULT_RUN}/plan.json')
    p.add_argument('--device-root', action='append')
    p.add_argument('--output')
    p.add_argument('--allow-partial', action='store_true')
    p = sub.add_parser('plots', help='Regenerate most-damaged beam-hinge graphs from verified saved trials; no analysis')
    p.add_argument('--plan', default=f'{PILOT}/prepared/{DEFAULT_RUN}/plan.json')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--device', type=int, choices=range(1, 5))
    source.add_argument('--device-root', action='append')
    p.add_argument('--output')
    p.add_argument('--member-scope', choices=('beam', 'all'), default='beam')
    args = parser.parse_args(argv)
    if args.action == 'setup':
        call([f'{PILOT}/verify_setup.py'])
    elif args.action in ('inventory', 'prepare'):
        prepare(args)
    elif args.action == 'run':
        run(args)
    elif args.action == 'plots':
        plots(args)
    else:
        collect(args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
