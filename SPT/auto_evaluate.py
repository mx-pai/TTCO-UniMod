#!/usr/bin/env python3
"""
Automatic evaluation script during training.
Evaluates checkpoints at specified epochs and records performance metrics.

Usage:
    python auto_evaluate.py --checkpoint_epoch 80 --config unimod1k_improved --save_results
"""
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

prj_path = os.path.dirname(__file__)
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Auto evaluate SPT checkpoints')
    parser.add_argument('--checkpoint_epoch', type=int, required=True, help='Epoch number to evaluate')
    parser.add_argument('--config', type=str, default='unimod1k_improved', help='Config name')
    parser.add_argument('--dataset', type=str, default='unimod1k', help='Dataset name')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads (0=sequential)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--save_results', action='store_true', help='Save evaluation results to JSON')
    parser.add_argument('--results_file', type=str, default='eval_history.json', help='Results JSON file')
    return parser.parse_args()


def run_evaluation(checkpoint_epoch, config, dataset, threads, num_gpus):
    """Run tracking evaluation"""
    print(f"\n{'='*80}")
    print(f"EVALUATING CHECKPOINT: epoch {checkpoint_epoch}")
    print(f"{'='*80}\n")

    # Update TEST.EPOCH in config
    config_path = os.path.join(prj_path, f'experiments/spt/{config}.yaml')
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        return None

    # Read and update config
    with open(config_path, 'r') as f:
        config_content = f.read()

    # Simple regex replacement for TEST.EPOCH
    import re
    config_content = re.sub(r'(TEST:\s*\n(?:.*\n)*?\s*EPOCH:\s*)(\d+)',
                           f'\\g<1>{checkpoint_epoch}', config_content)

    # Write to temp config
    temp_config_path = config_path.replace('.yaml', f'_temp_ep{checkpoint_epoch}.yaml')
    with open(temp_config_path, 'w') as f:
        f.write(config_content)

    print(f"[1/3] Updated config: TEST.EPOCH = {checkpoint_epoch}")

    # Run tracking
    cmd = [
        'python', 'tracking/test.py',
        '--tracker_name', 'spt',
        '--tracker_param', config.replace('_improved', ''),  # Use base config name
        '--dataset_name', dataset,
        '--runid', str(checkpoint_epoch),  # Use epoch as run_id
        '--threads', str(threads),
        '--num_gpus', str(num_gpus)
    ]

    print(f"[2/3] Running tracker: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=prj_path, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"[ERROR] Tracking failed:")
            print(result.stderr)
            return None
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("[ERROR] Tracking timeout (1 hour)")
        return None
    except Exception as e:
        print(f"[ERROR] Tracking failed: {e}")
        return None

    print(f"[3/3] Tracking completed")

    # Parse results (placeholder - actual implementation depends on your eval code)
    # You would typically parse the tracking results here
    results = {
        'epoch': checkpoint_epoch,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }

    # Cleanup temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    return results


def save_eval_results(results, results_file):
    """Append evaluation results to JSON file"""
    results_path = os.path.join(prj_path, results_file)

    # Load existing results
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            history = json.load(f)
    else:
        history = []

    # Append new results
    history.append(results)

    # Save
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Total evaluations: {len(history)}")


def find_best_checkpoint(results_file, metric='auc'):
    """Find the best checkpoint from evaluation history"""
    results_path = os.path.join(prj_path, results_file)

    if not os.path.exists(results_path):
        print("[WARN] No evaluation history found")
        return None

    with open(results_path, 'r') as f:
        history = json.load(f)

    if not history:
        return None

    # Find best based on metric (placeholder)
    best = max(history, key=lambda x: x.get(metric, 0))

    print(f"\n{'='*80}")
    print("BEST CHECKPOINT")
    print(f"{'='*80}")
    print(f"Epoch: {best['epoch']}")
    print(f"Metric ({metric}): {best.get(metric, 'N/A')}")
    print(f"{'='*80}\n")

    return best


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("SPT AUTO EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint epoch: {args.checkpoint_epoch}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*80}\n")

    # Set PYTHONPATH
    os.environ['PYTHONPATH'] = f"{prj_path}:{os.environ.get('PYTHONPATH', '')}"

    # Run evaluation
    results = run_evaluation(
        args.checkpoint_epoch,
        args.config,
        args.dataset,
        args.threads,
        args.num_gpus
    )

    if results is None:
        print("\n[ERROR] Evaluation failed")
        sys.exit(1)

    # Save results if requested
    if args.save_results:
        save_eval_results(results, args.results_file)

    # Show best checkpoint so far
    if args.save_results:
        find_best_checkpoint(args.results_file)

    print("\n✓ Evaluation completed successfully\n")


if __name__ == '__main__':
    main()

