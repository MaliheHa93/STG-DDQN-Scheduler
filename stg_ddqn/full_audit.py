"""Audit completed full-duration results without imposing a desired curve shape."""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import pandas as pd
from .config import load_config
from .runner import config_digest
from .validate_results import validate_raw,validate_summary


def audit(config,results,output):
    cfg=load_config(config)
    if cfg['simulation']['warmup_s']!=600 or cfg['simulation']['measurement_s']!=3600:
        raise ValueError('This audit requires the requested 600/3600-second protocol')
    if cfg['learning']['embedding_dimension']!=128 or cfg['learning']['training_episodes']!=15:
        raise ValueError('Compact validation models are not full paper-profile evidence')
    seeds=set(cfg['simulation']['seeds'])
    if len(seeds)!=10:raise ValueError('The full protocol requires ten evaluation seeds')
    results,output=Path(results),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(results/'evaluation_raw.csv')
    summary=pd.read_csv(results/'evaluation_summary.csv')
    validate_raw(raw);validate_summary(summary)
    axes=dict(traffic=cfg['experiments']['traffic_arrival_rates'],
              mobility=cfg['experiments']['mobility_speeds_m_s'],
              scalability=cfg['experiments']['scalability_nodes'])
    algorithms=set(cfg['experiments']['algorithms'])
    expected={(e,float(x),a,int(s)) for e,values in axes.items() for x in values
              for a in algorithms for s in seeds}
    actual=set(zip(raw.experiment,raw.x_value,raw.algorithm,raw.seed))
    if actual!=expected or len(raw)!=len(expected):
        raise ValueError(f'Full results incomplete: expected {len(expected)} unique runs, found {len(raw)}')
    if set(raw.mobility_mode)!={'sumo_csv'}:raise ValueError('Non-SUMO results in the full evaluation')
    for _,group in raw.groupby(['experiment','x_value','seed']):
        if group.sumo_trace_sha256.nunique()!=1 or group.generated_requests.nunique()!=1:
            raise ValueError('Comparison methods did not receive paired mobility/requests')
    marker=json.loads(Path(cfg['experiments']['checkpoint']+'.complete.json').read_text())
    from .train import _source_hashes
    if marker.get('source_sha256')!=_source_hashes():
        raise ValueError('Algorithm source differs from the completed training run')
    training_cfg=deepcopy(cfg)
    training_cfg['learning']['device']=marker['training_device']
    if marker['episodes']!=15 or marker['config_digest']!=config_digest(training_cfg):
        raise ValueError('The trained checkpoint does not match the full protocol')
    fingerprint=hashlib.sha256(Path(cfg['experiments']['checkpoint']).read_bytes()).hexdigest()
    if marker['checkpoint_sha256']!=fingerprint or set(raw[raw.algorithm.eq('stg_ddqn')].checkpoint_sha256)!={fingerprint}:
        raise ValueError('Learning results do not match the completed full-training checkpoint')
    rows=[]
    pattern=cfg['mobility']['sumo_trace_pattern']
    for cap in cfg['experiments']['mobility_speeds_m_s']:
        for seed in sorted(seeds):
            trace=Path(pattern.format(speed=f'{float(cap):g}',seed=seed))
            d=json.loads(trace.with_suffix('.json').read_text())
            if d['end_time_exclusive_s']<4200 or d['frames']<4200:
                raise ValueError('A SUMO trace is shorter than the full protocol')
            if d['csv_sha256']!=hashlib.sha256(trace.read_bytes()).hexdigest():
                raise ValueError('A SUMO trace changed since generation')
            population_samples=round(d['mean_concurrent_vehicles']*d['frames'])
            rows.append(dict(maximum_speed_m_s=cap,seed=seed,mean_speed_m_s=d['mean_speed_m_s'],
                observed_maximum_speed_m_s=d['observed_maximum_speed_m_s'],
                mean_population=d['mean_concurrent_vehicles'],unique_vehicles=d['unique_vehicles'],
                stopped_sample_fraction=d['stopped_vehicle_samples']/population_samples,
                trace_sha256=d['csv_sha256']))
    traces=pd.DataFrame(rows)
    traces.to_csv(output/'trace_statistics_by_seed.csv',index=False)
    traces.groupby('maximum_speed_m_s').agg(
        mean_speed_m_s=('mean_speed_m_s','mean'),sd_mean_speed_m_s=('mean_speed_m_s','std'),
        mean_population=('mean_population','mean'),mean_unique_vehicles=('unique_vehicles','mean'),
        mean_stopped_sample_fraction=('stopped_sample_fraction','mean')).reset_index().to_csv(output/'actual_speed_conditions.csv',index=False)
    mobility=raw[raw.experiment.eq('mobility')]
    mobility.to_csv(output/'continuity_by_seed.csv',index=False)
    metrics=['generated_requests','admitted_requests','continuity_resolved_requests',
        'continuity_satisfied_requests','continuity_failed_requests','active_excluded_requests',
        'vehicle_departure_requests','lifetime_expiry_requests','coverage_gap_failures',
        'search_budget_continuity_failures','active_service_seconds']
    counts=mobility.groupby(['x_value','algorithm'])[metrics].sum().reset_index()
    counts['censored_fraction_of_admitted']=counts.active_excluded_requests/counts.admitted_requests
    counts['pooled_continuity_pct']=100*counts.continuity_satisfied_requests/counts.continuity_resolved_requests
    counts['pooled_failures_per_service_hour']=3600*counts.continuity_failed_requests/counts.active_service_seconds
    counts.to_csv(output/'continuity_counts_by_point.csv',index=False)
    requests=pd.read_csv(results/'request_records.csv')
    tracked=requests[requests.tracked.eq(True)]
    tracked.groupby(['experiment','x_value','algorithm','seed','departure_reason']).size().rename('count').reset_index().to_csv(output/'termination_reasons.csv',index=False)
    failures=tracked[tracked.continuity_failed.eq(True)]
    failures.groupby(['experiment','x_value','algorithm','departure_reason']).size().rename('count').reset_index().to_csv(output/'failure_reasons.csv',index=False)
    primary=summary[summary.experiment.eq('mobility') & summary.metric.eq('continuity_satisfaction_ratio_pct')]
    if not primary.n.eq(10).all():raise ValueError('Some continuity points lack ten defined seed-level ratios')
    primary.to_csv(output/'continuity_primary_means_and_intervals.csv',index=False)
    text=['# Full-duration SUMO continuity audit','',
        f'Completed main evaluations: {len(raw)}. Warmup/measurement: 600/3600 seconds.',
        'Primary curve: mean of ten seed-level continuity ratios, with the recorded Student-t interval.',
        'Pooled counts and exposure rates are supplementary and do not replace that estimand.',
        'An unbounded Student-t interval may extend beyond physical percentage limits.',
        'Zero observed failures does not establish perfect continuity or zero future failure risk.','',
        '## Counts and censoring','```csv\n'+counts.to_csv(index=False)+'```',
        '## Actual speeds',traces.groupby('maximum_speed_m_s').mean(numeric_only=True).to_csv(),
        '## Interpretation','A decline or rebound must be explained using these counts, termination reasons, actual speeds and exposure rates. No monotonic trend is imposed.',
        'The generated grid is a reproducible benchmark, not the unavailable original SUMO scenario.',
        'Reference timing is included; controller computation does not advance physical simulation time.']
    (output/'FULL_RESULTS_AUDIT.md').write_text('\n\n'.join(text)+'\n')
    return dict(status='complete',main_evaluation_runs=len(raw),trace_conditions=len(traces),
                total_mobility_resolved=int(counts.continuity_resolved_requests.sum()),
                total_mobility_failures=int(counts.continuity_failed_requests.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/sumo_paper.yaml')
    p.add_argument('--results',default='results/sumo_paper')
    p.add_argument('--output',default='results/sumo_paper/audit')
    a=p.parse_args();print(json.dumps(audit(a.config,a.results,a.output),indent=2))


if __name__=='__main__':main()
