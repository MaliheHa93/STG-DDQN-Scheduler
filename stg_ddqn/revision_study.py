"""Retrained reward, arrival-model and temporal-encoder comparisons.

Each summary sample is a training replicate averaged over paired evaluation
seeds. Evaluation repetitions of one checkpoint are not independent training
replicates. All jobs run sequentially to avoid contaminating runtime data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pandas as pd
import yaml

from .config import deep_merge, load_config
from .runner import run_single,config_digest
from .mobility import resolve_sumo_run
from .train import _source_hashes
from .stats import PAPER_METRICS, summarize_results


def variants(study):
    if study == 'reward':
        return [(name, 'stg_ddqn', {'reward': dict(zip(
            ('latency', 'lifetime_benefit', 'migration_volume', 'downtime'), weights))})
            for name, weights in [
                ('original', (.35, .25, .25, .15)),
                ('delay_oriented', (.50, .30, .15, .05)),
                ('migration_oriented', (.20, .15, .45, .20)),
                ('equal', (.25, .25, .25, .25)),
            ]]
    if study == 'bursty':
        return [(name, 'stg_ddqn', {'traffic': {'arrival_model': name}})
                for name in ('poisson', 'mmpp')]
    return [('spatial_temporal', 'stg_ddqn', {}), ('spatial_only', 'graph_ddqn', {})]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sumo_paper.yaml')
    parser.add_argument('--study', choices=('reward', 'bursty', 'ablation'), required=True)
    parser.add_argument('--training-seeds', default='10001,20001,30001')
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-device',choices=('cpu','cuda','mps'),default=None)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    source = Path(args.config).resolve()
    cfg = load_config(source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path=output/'evaluation_raw.csv'
    request_path=output/'request_records.csv'
    raw=pd.read_csv(raw_path).to_dict('records') if args.resume and raw_path.exists() else []
    requests=pd.read_csv(request_path).to_dict('records') if args.resume and request_path.exists() else []
    evaluation_seeds = list(cfg['simulation']['seeds'])
    training_seeds = [int(x) for x in args.training_seeds.split(',')]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError('Training seeds must be distinct')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for index, (name, algorithm, override) in enumerate(variants(args.study), 1):
        for training_seed in training_seeds:
            directory = output / name / str(training_seed)
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / 'config.yaml'
            run_override = deep_merge(override, {'simulation': {'seeds': [training_seed]}})
            run_override['training_seed'] = training_seed
            run_override['extends'] = os.path.relpath(source, directory)
            config_path.write_text(yaml.safe_dump(run_override), encoding='utf-8')
            checkpoint = directory / 'checkpoint.pt'
            training_cfg=load_config(config_path)
            if args.training_device:training_cfg['learning']['device']=args.training_device
            cache_key=hashlib.sha256(json.dumps(dict(config=config_digest(training_cfg),algorithm=algorithm,
                         source=_source_hashes()),sort_keys=True).encode()).hexdigest()
            cache=output.parent/'.trained_model_cache'/cache_key
            if args.resume and not checkpoint.exists() and (cache/'checkpoint.pt').exists():
                for name in ('checkpoint.pt','checkpoint.pt.complete.json','training_log.csv'):
                    if (cache/name).exists():shutil.copy2(cache/name,directory/name)
            command=[sys.executable, '-u', '-m', 'stg_ddqn.train',
                    '--config', str(config_path), '--policy', algorithm,
                    '--checkpoint', str(checkpoint), '--output', str(directory / 'training')]
            if args.training_device:command+=['--device',args.training_device]
            if args.resume:command+=['--resume']
            with (directory / 'training_stdout.log').open('a' if args.resume else 'w') as log:
                subprocess.run(command,
                    env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
            cache.mkdir(parents=True,exist_ok=True)
            for name in ('checkpoint.pt','checkpoint.pt.complete.json'):
                shutil.copy2(directory/name,cache/name)
            if (directory/'training/training_log.csv').exists():
                shutil.copy2(directory/'training/training_log.csv',cache/'training_log.csv')
            evaluation_cfg = load_config(config_path)
            evaluation_cfg['simulation']['seeds'] = evaluation_seeds
            for seed in evaluation_seeds:
                existing=[r for r in raw if r['algorithm']==name and int(r['training_seed'])==training_seed
                          and int(r['seed'])==int(seed)]
                fingerprint=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if existing:
                    if len(existing)!=1 or existing[0]['checkpoint_sha256']!=fingerprint:
                        raise ValueError('Study checkpoint changed or duplicate evaluation key')
                    resolved=resolve_sumo_run(evaluation_cfg,int(seed))
                    if (existing[0]['config_digest']!=config_digest(resolved) or
                        existing[0]['sumo_trace_sha256']!=hashlib.sha256(Path(resolved['mobility']['sumo_csv']).read_bytes()).hexdigest()):
                        raise ValueError('Study configuration or SUMO trace changed')
                    continue
                row, records, _ = run_single(evaluation_cfg, algorithm, int(seed), checkpoint=str(checkpoint))
                metadata = dict(experiment=args.study, x_name='variant_index', x_value=index,
                    algorithm=name, training_seed=training_seed,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                row.update(metadata)
                raw.append(row)
                for record in records:
                    record.update(metadata)
                    requests.append(record)
                pd.DataFrame(raw).to_csv(raw_path,index=False)
                pd.DataFrame(requests).to_csv(request_path,index=False)
                (output/'progress.json').write_text(json.dumps(dict(status='evaluating',variant=name,
                    training_seed=training_seed,evaluation_seed=int(seed),stored_runs=len(raw)),indent=2)+'\n')
            pd.DataFrame(raw).to_csv(output / 'evaluation_raw.csv', index=False)
            pd.DataFrame(requests).to_csv(output / 'request_records.csv', index=False)
            print(f'{args.study}: {name}, training seed {training_seed} complete', flush=True)
    frame = pd.DataFrame(raw)
    group = ['experiment', 'x_name', 'x_value', 'algorithm', 'training_seed']
    metrics = [metric for metric in PAPER_METRICS if metric in frame]
    replicates = frame.groupby(group, as_index=False)[metrics].mean()
    # A short run can have no resolved services. Keep undefined ratios as NaN,
    # and disclose how many evaluation realizations contribute to each mean.
    frame.groupby(group, as_index=False)[metrics].count().to_csv(
        output / 'evaluation_metric_counts.csv', index=False)
    replicates.to_csv(output / 'training_replicate_means.csv', index=False)
    summarize_results(replicates).to_csv(output / 'evaluation_summary.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps({
        'study': args.study, 'training_seeds': training_seeds,
        'evaluation_seeds': evaluation_seeds, 'mobility_mode': cfg['mobility']['mode'],
        'ci_unit': 'independent training replicate mean over paired evaluation seeds',
        'scenario': 'fixed configured load and speed; no speed sweep',
        'undefined_metrics': 'available-case means; per-metric evaluation counts in evaluation_metric_counts.csv',
        'training_device':args.training_device or cfg['learning']['device'],
        'identical_condition_reuse':'identical seed/configuration/algorithm/source checkpoints reused across studies',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
