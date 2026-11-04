"""Generate real SUMO road simulations and epoch CSVs, with provenance.

This is a reproducible benchmark scenario, not a reconstruction of an
unavailable original SUMO map. All randomness is explicitly seeded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .config import load_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fcd_to_csv(source, destination, offset, duration, epoch, maximum_speed):
    """Stream FCD output; empty timesteps remain represented in the metadata."""
    counts, speeds, unique = [], [], set()
    expected = int(round(duration / epoch))
    frames = set()
    with Path(destination).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('time_s', 'vehicle_id', 'x_m', 'y_m', 'speed_m_s', 'heading_rad'))
        for _, element in ET.iterparse(source, events=('end',)):
            if element.tag != 'timestep':
                continue
            time = float(element.attrib['time']) - offset
            frame = int(round(time / epoch))
            if -1e-8 <= time < duration - 1e-8:
                if abs(time / epoch - frame) > 1e-6:
                    raise ValueError('SUMO output is not aligned to the requested epoch')
                frames.add(frame)
                counts.append(len(element))
                for vehicle in element:
                    a = vehicle.attrib
                    speed = float(a['speed'])
                    if speed > maximum_speed + .02:
                        raise ValueError('SUMO exceeded the configured vehicle speed cap')
                    writer.writerow((frame * epoch, a['id'], a['x'], a['y'], speed,
                                     np.deg2rad(90.0 - float(a['angle']))))
                    unique.add(a['id'])
                    speeds.append(speed)
            element.clear()
    if frames != set(range(expected)):
        raise ValueError('SUMO output does not cover every requested epoch')
    if not speeds:
        raise ValueError('SUMO generated no observed vehicles')
    return dict(start_time_s=0, end_time_exclusive_s=duration, epoch_s=epoch,
                frames=expected, maximum_concurrent_vehicles=max(counts),
                mean_concurrent_vehicles=float(np.mean(counts)), unique_vehicles=len(unique),
                observed_maximum_speed_m_s=max(speeds), mean_speed_m_s=float(np.mean(speeds)),
                stopped_vehicle_samples=sum(speed < .1 for speed in speeds),
                csv_sha256=digest(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sumo_paper.yaml')
    parser.add_argument('--seeds', help='Override generation seeds, comma-separated')
    parser.add_argument('--speeds', help='Override speed caps, comma-separated')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    mobility = cfg['mobility']
    pattern = mobility['sumo_trace_pattern']
    seeds = ([int(x) for x in args.seeds.split(',')] if args.seeds else
             list(dict.fromkeys(mobility['evaluation_trace_seeds'] + mobility['training_trace_seeds'])))
    speeds = ([float(x) for x in args.speeds.split(',')] if args.speeds else
              cfg['experiments']['mobility_speeds_m_s'])
    try:
        import sumo
    except ImportError as exc:
        raise SystemExit('Install requirements-sumo.txt before generating scenarios') from exc
    home = Path(sumo.SUMO_HOME)
    environment = dict(os.environ, SUMO_HOME=str(home), PATH=str(home / 'bin') + os.pathsep + os.environ['PATH'])
    executable = lambda name: str(home / 'bin' / name)
    version = subprocess.check_output([executable('sumo'), '--version'], env=environment, text=True).splitlines()[0]
    duration = float(cfg['simulation']['warmup_s']) + float(cfg['simulation']['measurement_s'])
    epoch = float(cfg['simulation']['epoch_s'])
    preroll = float(mobility.get('sumo_preroll_s', 120))
    total = duration + preroll
    for speed in speeds:
        for seed in seeds:
            if (not args.seeds and seed in mobility['training_trace_seeds'] and
                    float(speed) != float(mobility['maximum_speed_m_s'])):
                continue
            destination = Path(pattern.format(speed=f'{speed:g}', seed=seed))
            directory = destination.parent
            directory.mkdir(parents=True, exist_ok=True)
            metadata_path = destination.with_suffix('.json')
            recipe = dict(speed_m_s=float(speed), seed=seed, duration_s=duration,
                          epoch_s=epoch, preroll_s=preroll, area_m=cfg['network']['area_m'],
                          vehicle_cap=int(mobility['maximum_vehicles']),
                          departure_period_s=float(mobility.get('sumo_departure_period_s', .3)),
                          generator='grid5x5_randomTrips_intermediate2_v1', sumo_version=version)
            if args.resume and destination.exists() and metadata_path.exists():
                previous = json.loads(metadata_path.read_text())
                if previous.get('recipe') == recipe and previous.get('csv_sha256') == digest(destination):
                    print(f'resume {destination}', flush=True)
                    continue
            commands = []
            def run(command, log):
                commands.append(command)
                with (directory / log).open('w') as out:
                    subprocess.run(command, env=environment, stdout=out, stderr=subprocess.STDOUT, check=True)
            width, height = cfg['network']['area_m']
            run([executable('netgenerate'), '--grid', '--grid.x-number', '5', '--grid.y-number', '5',
                 '--grid.x-length', str(width / 4), '--grid.y-length', str(height / 4),
                 '--default.speed', str(speed), '--default.lanenumber', '2',
                 '--tls.guess', 'true', '--seed', str(seed), '-o', str(directory / 'roads.net.xml')], 'netgenerate.log')
            root = ET.Element('additional')
            ET.SubElement(root, 'vType', id='car', vClass='passenger', maxSpeed=str(speed),
                          speedFactor='1', speedDev='0', carFollowModel='Krauss',
                          accel='2.6', decel='4.5', sigma='0.5', length='5', minGap='2.5')
            ET.ElementTree(root).write(directory / 'types.add.xml')
            run([sys.executable, str(home / 'tools' / 'randomTrips.py'),
                 '-n', str(directory / 'roads.net.xml'), '-a', str(directory / 'types.add.xml'),
                 '-o', str(directory / 'trips.trips.xml'), '-r', str(directory / 'routes.rou.xml'),
                 '--seed', str(seed), '-b', '0', '-e', str(total),
                 '-p', str(recipe['departure_period_s']), '--intermediate', '2',
                 '--min-distance', '600', '--trip-attributes', 'type="car" departLane="best" departSpeed="random"',
                 '--threads', '1'], 'routing.log')
            root = ET.Element('configuration')
            inputs = ET.SubElement(root, 'input')
            # duarouter already embeds the car vType in routes.rou.xml.
            for key, value in [('net-file', 'roads.net.xml'), ('route-files', 'routes.rou.xml')]:
                ET.SubElement(inputs, key, value=value)
            ET.ElementTree(root).write(directory / 'scenario.sumocfg')
            fcd = directory / 'trajectory.fcd.xml'
            run([executable('sumo'), '-c', str(directory / 'scenario.sumocfg'), '--seed', str(seed),
                 '--step-length', str(epoch), '--end', str(total), '--no-step-log', 'true',
                 '--time-to-teleport', '-1', '--max-num-vehicles', str(mobility['maximum_vehicles']),
                 '--fcd-output', str(fcd), '--precision', '6',
                 '--summary-output', str(directory / 'traffic_summary.xml')], 'sumo.log')
            metadata = fcd_to_csv(fcd, destination, preroll, duration, epoch, speed)
            metadata.update(recipe=recipe, commands=commands,
                            network_sha256=digest(directory / 'roads.net.xml'),
                            routes_sha256=digest(directory / 'routes.rou.xml'))
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            fcd.unlink()  # CSV, scenario, routes and metadata are sufficient to reproduce it.
            print(f'generated {destination}: {metadata["frames"]} frames, '
                  f'mean vehicles={metadata["mean_concurrent_vehicles"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
