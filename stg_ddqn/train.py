from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import atexit
from pathlib import Path

import pandas as pd

from .config import load_config
from .policies import make_policy
from .runner import config_digest, run_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STG-DDQN with delayed migration transitions")
    parser.add_argument("--config", default="configs/sumo_paper.yaml")
    parser.add_argument("--policy", choices=("stg_ddqn", "graph_ddqn"), default="stg_ddqn")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/training")
    parser.add_argument('--device', choices=('cpu','cuda','mps'), default=None)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Resume only a trusted local training state')
    parser.add_argument('--checkpoint-interval-s', type=int, default=300)
    parser.add_argument('--progress-interval-s', type=int, default=60)
    args = parser.parse_args()
    if args.checkpoint_interval_s <= 0 or args.progress_interval_s <= 0:
        parser.error('Checkpoint and progress intervals must be positive')
    cfg = load_config(args.config)
    if args.device:
        cfg['learning']['device'] = args.device
    import torch
    torch.set_num_threads(max(1,args.threads))
    checkpoint = Path(
        args.checkpoint or cfg["experiments"]["checkpoint"] or "checkpoints/stg_ddqn.pt"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock,Timeout
    training_lock=FileLock(str(checkpoint)+'.lock')
    try:
        training_lock.acquire(timeout=0)
    except Timeout:
        if not args.resume:
            raise RuntimeError('This checkpoint is already being trained by another process')
        print('Waiting for the existing training process to finish...',flush=True)
        training_lock.acquire()
    atexit.register(training_lock.release)
    seeds = [int(seed) for seed in cfg["mobility"].get('training_trace_seeds', cfg["simulation"]["seeds"])]
    if cfg.get('training_seed') is not None:
        # Study replicates independently initialize learning and request streams.
        # Their configured mobility seed remains in the training-only pool.
        initialization_seed = int(cfg['training_seed'])
    else:
        initialization_seed = seeds[0]
    completion_path = checkpoint.with_name(checkpoint.name + '.complete.json')
    resume_path = checkpoint.with_name(checkpoint.name + '.resume.pt')
    if args.resume and completion_path.exists() and checkpoint.exists():
        completed = json.loads(completion_path.read_text())
        if (completed.get('config_digest') == config_digest(cfg) and
                completed.get('checkpoint_sha256') == _sha256(checkpoint) and
                completed.get('source_sha256') == _source_hashes()):
            print(f'completed training checkpoint verified: {checkpoint}',flush=True)
            return
        raise ValueError('Existing completed model differs from this run; select a new checkpoint path')
    policy = make_policy(args.policy, cfg, seed=initialization_seed + 10_000, training=True)
    rows = []
    episodes = int(cfg["learning"]["training_episodes"])
    partial_checkpoint = checkpoint.with_name(checkpoint.name + ".partial")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    first_episode, next_frame, resume_environment, prior_wall_s = 0, 0, None, 0.0
    if args.resume and resume_path.exists():
        # This is an internal state created by this program, not an imported
        # model. Loading arbitrary pickled checkpoints is intentionally not a
        # supported interchange workflow.
        marker = json.loads(resume_path.with_suffix('.json').read_text())
        if marker['sha256'] != _sha256(resume_path):
            raise ValueError('Training resume-state checksum mismatch')
        state = torch.load(resume_path, map_location=policy.device, weights_only=False)
        if state['config_digest'] != config_digest(cfg) or state['source_sha256'] != _source_hashes():
            raise ValueError('Training configuration or algorithm source changed since the saved state')
        policy, resume_environment = state['policy'], state['environment']
        rows, first_episode, next_frame = state['rows'], state['episode'], state['next_frame']
        prior_wall_s = state.get('wall_s',0)
        torch.set_rng_state(state['torch_rng'].cpu())
        if state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'].cpu())
        if state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        print(f'resuming episode={first_episode+1} next_frame={next_frame}',flush=True)
    started = time.monotonic()
    def save_resume(episode, environment, frame):
        state = dict(config_digest=config_digest(cfg), source_sha256=_source_hashes(),
            policy=policy, environment=environment, episode=episode,next_frame=frame,rows=rows,
            wall_s=prior_wall_s+time.monotonic()-started,
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if policy.device.type=='mps' else None,
            cuda_rng=torch.cuda.get_rng_state_all() if policy.device.type=='cuda' else None)
        temporary = resume_path.with_name(resume_path.name+'.tmp')
        torch.save(state,temporary)
        os.replace(temporary,resume_path)
        _write_json(resume_path.with_suffix('.json'),dict(sha256=_sha256(resume_path),
                    episode=episode,next_frame=frame,wall_s=state['wall_s']))
    for episode in range(first_episode,episodes):
        seed = seeds[episode % len(seeds)] + 100_000 * (episode // len(seeds))
        epoch_s=float(cfg['simulation']['epoch_s'])
        def callback(environment, frame, total):
            sample_s=frame*epoch_s
            if frame == total or sample_s % args.progress_interval_s < epoch_s:
                tracked=[r for r in environment.records.values() if r.tracked]
                progress=dict(status='training',episode=episode+1,episodes=episodes,
                    next_frame=frame,total_frames=total,simulated_s=sample_s,
                    gradient_updates=policy.update_count,replay_transitions=len(policy.replay),
                    admitted=sum(r.admitted for r in tracked),resolved=sum(r.continuity_resolved for r in tracked),
                    failed=sum(r.continuity_failed for r in tracked),device=str(policy.device),
                    wall_s=prior_wall_s+time.monotonic()-started)
                _write_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            if sample_s % args.checkpoint_interval_s < epoch_s:
                save_resume(episode,environment,frame)
        row, _, policy = run_single(
            cfg, args.policy, seed, training=True, policy=policy,
            environment=resume_environment,start_frame=next_frame,epoch_callback=callback,
        )
        resume_environment, next_frame = None, 0
        row["episode"] = episode + 1
        rows.append(row)
        # Keep the public checkpoint immutable until every configured episode
        # has finished.  Evaluation can therefore never load a mixture of
        # weights from an in-progress training run.
        policy.save(partial_checkpoint)
        pd.DataFrame(rows).to_csv(output / "training_log.csv", index=False)
        save_resume(episode+1,None,0)
        print(
            f"episode={episode + 1}/{episodes} seed={seed} "
            f"acceptance={row['acceptance_ratio_pct']:.2f}% "
            f"continuity={row['continuity_satisfaction_ratio_pct']:.2f}% "
            f"reward={row['mean_reward']:.4f}",flush=True
        )
    os.replace(partial_checkpoint, checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    completion = {
        "status": "complete",
        "episodes": episodes,
        "config_digest": config_digest(cfg),
        "checkpoint_sha256": checkpoint_hash,
        'source_sha256': _source_hashes(), 'training_device': str(policy.device),
        'training_threads':args.threads,'gradient_updates':policy.update_count,
    }
    marker = checkpoint.with_name(checkpoint.name + ".complete.json")
    marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    _write_json(output/'progress.json',completion)
    resume_path.unlink(missing_ok=True)
    resume_path.with_suffix('.json').unlink(missing_ok=True)
    print(f"checkpoint={checkpoint.resolve()}")
    print(f"checkpoint_sha256={checkpoint_hash}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes():
    directory=Path(__file__).parent
    names=('train','neural','environment','runner','config','mobility','topology',
           'candidates','state','types','requests','policies','reward')
    return {name:_sha256(directory/(name+'.py')) for name in names}


def _write_json(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temporary,path)


if __name__ == "__main__":
    main()
