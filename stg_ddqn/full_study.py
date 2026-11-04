"""Durable full-duration study: never replace the paper profile with a short run."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from filelock import FileLock
from .config import load_config
from .full_audit import audit

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/sumo_paper.yaml')
    parser.add_argument('--training-device',choices=('auto','cpu','cuda','mps'),default='auto')
    parser.add_argument('--output',default='results/sumo_paper')
    parser.add_argument('--delivery',default='../output/SUMO_FULL_DURATION')
    args=parser.parse_args()
    os.chdir(ROOT)
    cfg=load_config(args.config)
    if (cfg['simulation']['warmup_s'],cfg['simulation']['measurement_s'],
        cfg['learning']['embedding_dimension'],cfg['learning']['training_episodes'])!=(600,3600,128,15):
        raise ValueError('Full workflow requires the unshortened paper profile')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    delivery=Path(args.delivery).resolve();delivery.mkdir(parents=True,exist_ok=True)
    import torch
    device=args.training_device
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    status_path=output/'full_study_status.json'
    status=dict(status='running',pid=os.getpid(),started_request_date='2026-11-04',
        profile=str(Path(args.config).resolve()),training_device=device,
        expected_main_runs=450,expected_extended_runs=240,
        training_progress=str(ROOT/'results/sumo_paper_training/progress.json'),
        evaluation_progress=str(output/'progress.json'),delivery=str(delivery),completed_stages=[])
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTORCH_ENABLE_MPS_FALLBACK='1',STG_RUNTIME_CONTEXT='sequential_after_full_training',
        MPLCONFIGDIR='/private/tmp/sfc-full-matplotlib')
    python=sys.executable
    logdir=output/'logs';logdir.mkdir(exist_ok=True)
    checkpoint=str(Path(cfg['experiments']['checkpoint']))
    def run(stage,command):
        status.update(status='running',stage=stage,command=command,last_stage_started_unix=time.time())
        write_json(status_path,status)
        print(json.dumps(dict(stage=stage,status='started')),flush=True)
        with (logdir/(stage+'.log')).open('a') as log:
            subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        if stage not in status['completed_stages']:status['completed_stages'].append(stage)
        write_json(status_path,status)
    with FileLock(str(output/'.full_study.lock'),timeout=0):
        try:
            run('generate_full_sumo',[python,'-u','-m','stg_ddqn.sumo_scenario','--config',args.config,'--resume'])
            run('train_full_model',[python,'-u','-m','stg_ddqn.train','--config',args.config,
                '--device',device,'--resume','--checkpoint',checkpoint,
                '--output',str(output.with_name(output.name+'_training'))])
            # Existing heuristic writers hold the evaluation lock. This call
            # waits and then resumes their scientific rows. Scalability timing
            # begins after training and runs without another study worker.
            run('evaluate_full_main',[python,'-u','-m','stg_ddqn.experiments','--config',args.config,
                '--checkpoint',checkpoint,'--output',str(output),'--resume','--workers','1'])
            assessment=audit(args.config,output,output/'audit')
            write_json(output/'FULL_MAIN_COMPLETE.json',assessment)
            run('render_full_main_python',[python,'tools/render_sumo_previews.py','--full',
                '--config',args.config,'--results',str(output),'--output',str(delivery/'plots_python')])
            for study in ('reward','bursty','ablation'):
                run('study_'+study,[python,'-u','-m','stg_ddqn.revision_study','--config',args.config,
                    '--training-device',device,'--resume','--study',study,'--output',str(output/study)])
            import pandas as pd
            for study,count in [('reward',120),('bursty',60),('ablation',60)]:
                frame=pd.read_csv(output/study/'evaluation_raw.csv')
                if len(frame)!=count or frame.duplicated(['algorithm','training_seed','seed']).any():
                    raise ValueError('An extended study is incomplete or contains duplicate evaluations')
                if set(frame.mobility_mode)!={'sumo_csv'}:
                    raise ValueError('Extended study includes non-SUMO data')
            write_json(output/'FULL_STUDY_COMPLETE.json',dict(status='complete',
                main_runs=450,extended_runs=240,training_device=device,assessment=assessment))
            run('render_all_full_python',[python,'tools/render_sumo_previews.py','--full',
                '--config',args.config,'--results',str(output),'--output',str(delivery/'plots_python')])
            run('package_full_delivery',[python,'tools/package_full_delivery.py',
                '--results',str(output),'--delivery',str(delivery)])
            status.update(status='complete',stage='finished',finished_unix=time.time(),
                matlab='Code supplied; figures here rendered using Python, not MATLAB')
            write_json(status_path,status)
        except BaseException as error:
            status.update(status='failed_or_interrupted',error=repr(error),traceback=traceback.format_exc())
            write_json(status_path,status)
            raise


if __name__=='__main__':main()
