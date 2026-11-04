"""Python previews of measured SUMO validation data; not MATLAB output."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from stg_ddqn.plot_results import PANELS, _configure_style, _draw_panel, validate_curve_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--results',default='results/sumo_validation')
    parser.add_argument('--config',default='configs/sumo_paper.yaml')
    parser.add_argument('--full',action='store_true',help='Require the completed full paper protocol')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.results)
    if not source.is_absolute():source=ROOT/source
    if args.full:
        from stg_ddqn.full_audit import audit
        audit(ROOT/args.config,source,source/'audit')
    table = pd.read_csv(source / 'evaluation_summary.csv')
    validate_curve_data(table)
    _configure_style()
    main_names = ['request_acceptance','sfc_continuity','migration_volume','decision_runtime']
    panels = list(zip(main_names, PANELS))
    for name, exp, metric, xlabel, ylabel in [
        ('mobility_failure_rate','mobility','failures_per_service_hour','Maximum vehicle speed (m/s)','Failures per service-hour'),
        ('vehicle_departures','mobility','vehicle_departure_requests','Maximum vehicle speed (m/s)','SFCs resolved by vehicle departure'),
        ('coverage_failures','mobility','coverage_gap_failures','Maximum vehicle speed (m/s)','Coverage-gap failures'),
        ('censored_sfcs','mobility','active_excluded_requests','Maximum vehicle speed (m/s)','SFCs censored at measurement end'),
        ('temporary_cpu_peak','traffic','peak_temporary_cpu','Arrival rate $\\lambda$ (requests/s)','Peak temporary CPU reservation (model units)'),
        ('temporary_cpu_integral','traffic','temporary_cpu_seconds','Arrival rate $\\lambda$ (requests/s)','Temporary CPU reservation (model unit s)'),
        ('reservation_audit','traffic','blocked_recoverable_without_temporary_reservations','Arrival rate $\\lambda$ (requests/s)','Admissions recovered in snapshot audit'),
        ('search_budget_blocks','traffic','search_budget_blocked_requests','Arrival rate $\\lambda$ (requests/s)','Blocks after search-budget exhaustion'),
        ('candidate_runtime','scalability','mean_candidate_runtime_ms','Number of fog RSUs','Mean candidate construction runtime (ms)'),
        ('policy_runtime','scalability','mean_policy_runtime_ms','Number of fog RSUs','Mean policy selection runtime (ms)'),
        ('reference_route_runtime','scalability','mean_reference_route_runtime_ms','Number of fog RSUs','Mean reference-route runtime (ms)'),
        ('assignment_search_work','scalability','assignment_nodes_examined','Number of fog RSUs','Partial assignment trials during measurement'),
    ]:
        panels.append((name,(exp,metric,xlabel,ylabel,'best')))
    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    for name, panel in panels:
        fig, ax = plt.subplots(figsize=(7.2,5),layout='constrained')
        _draw_panel(ax, table, panel, show_ci=True)
        low, high = ax.get_ylim()
        span = high - low
        ax.set_ylim(low - .04 * span, high + .06 * span)
        save(fig, output / name)
    for study in ('reward','bursty','ablation'):
        if not (source/study/'evaluation_summary.csv').exists():
            continue
        table = pd.read_csv(source / study / 'evaluation_summary.csv')
        for metric, label in [
            ('acceptance_ratio_pct','Request acceptance (%)'),
            ('continuity_satisfaction_ratio_pct','SFC continuity satisfaction (%)'),
            ('migration_mb_per_admitted_sfc','Transferred payload per admitted SFC (MB)'),
            ('failures_per_service_hour','Failures per service-hour'),
        ]:
            rows = table[table.metric.eq(metric)].sort_values('x_value')
            assert rows.n.ge(2).all()
            fig, ax = plt.subplots(figsize=(8,5),layout='constrained')
            ax.errorbar(rows.x_value,rows['mean'],
                yerr=[rows['mean']-rows.ci95_low,rows.ci95_high-rows['mean']],
                fmt='o',color='#0072B2',capsize=4)
            ax.set_xticks(rows.x_value, rows.algorithm.str.replace('_',' '),rotation=15)
            ax.set_ylabel(label)
            ax.grid(True)
            save(fig, output / study / metric)
    description=(
        'REAL SUMO FULL PAPER PROTOCOL — PYTHON RENDERING, NOT MATLAB OUTPUT\n'
        '600 s scheduler warmup, 3600 s measurement, 10 paired evaluation seeds.\n'
        'Full 128-dimensional model, 15 training episodes.\n'
        'Main intervals describe evaluation variation conditional on one trained model.\n'
        'Study intervals describe independent training-replicate means.\n'
        'Use the source audit for failures, censoring, departures and actual speed distributions.\n'
    ) if args.full else (
        'REAL SUMO SHORT VALIDATION — PYTHON PREVIEWS, NOT MATLAB RENDERS\n'
        'Main sweep: 135 runs, 3 paired evaluation seeds, one trained policy.\n'
        'Extended studies: 24 independently trained models and 72 evaluations.\n'
        '10 s scheduler warmup, 80 s measurement, compact encoder, 3 training episodes.\n'
        'Not final paper-scale or convergence evidence.\n'
    )
    (output / 'README.txt').write_text(description+
        'Error bars: recorded 95% Student-t confidence intervals. Main intervals\n'
        'are across evaluation seeds; study intervals are across training replicate means.\n'
        'Unbounded intervals may extend outside physical ratio/count ranges.\n'
        'No top titles or panel letters. No smoothing or replacement of measurements.\n'
        'For MATLAB rendering use matlab/RUN_SUMO_PLOTS.m in the full code package.\n')
    paths = [source / 'evaluation_summary.csv'] + [source / s / 'evaluation_summary.csv' for s in ('reward','bursty','ablation')]
    (output / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file(): z.write(p,Path(output.name)/p.relative_to(output))
    print(archive)


if __name__ == '__main__':
    main()
