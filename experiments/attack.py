from typing import Optional, Dict, get_args
from pathlib import Path
import argparse
import logging

import numpy as np
from numpy.typing import NDArray
from lightning import Trainer
from uncropped_transformers.datasets import DATASET
from uncropped_transformers.evaluation.mtd import accumulate_ranks, compute_mtd

from init_things import *
from init_things.directories import append_directory_clargs, init_directories, load_directory_config
from utils.load_things import load_numpy_dataset, load_torch_dataset, construct_loaders, load_trained_model
from utils.training_config import SupervisedTrainingConfig
from utils.artifacts import artifact_identity, metadata_json, cache_matches, summarize_ranks


def run_attack_performance_evalutaion(ckpt_path: Path, dataset_id: DATASET, dataset_kwargs: Optional[Dict] = None) -> Dict[str, NDArray[np.floating]]:
    dataset_kwargs = dataset_kwargs or {}

    attack_set = load_torch_dataset(dataset_id, 'attack', **dataset_kwargs)
    attack_loader, = construct_loaders([], [attack_set])
    module = load_trained_model(ckpt_path, None)
    module.to('cuda')
    trainer = Trainer(
        accelerator='gpu',
        precision='bf16-mixed',
        default_root_dir=None,
        logger=False
    )
    results = trainer.test(module, dataloaders=attack_loader)
    metrics = {k: np.array(v) for k, v in results[0].items()}

    # Retain every attack's rank trajectory, plus per-output summaries.
    outputs = trainer.predict(module, dataloaders=attack_loader)
    all_byte_logits = torch.cat([o[0] for o in outputs])
    all_int_vars = {k: torch.cat([o[1][k] for o in outputs]) for k in outputs[0][1]}
    mtd_kwargs = module.config.mtd_kwargs
    rank_over_time = accumulate_ranks(
        all_byte_logits, all_int_vars,
        mtd_kwargs['target_preds_to_key_preds'],
        attack_count=mtd_kwargs['attack_count'],
        traces_per_attack=mtd_kwargs.get('traces_per_attack')
    )
    # rank_over_time: [num_attacks, num_traces, num_bytes]
    metrics.update(summarize_ranks(rank_over_time))
    metrics['per_byte_mtd'] = compute_mtd(rank_over_time, reduction='none').mean(axis=0)  # [num_bytes]
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compute and cache checkpoint performance on the attack set.')
    parser.add_argument(
        '--model-ckpt-path', required=True, type=Path,
        help='Checkpoint to evaluate; config.yaml is detected beside it.'
    )

    parser.add_argument(
        '--dataset', default=None, choices=get_args(DATASET),
        help='Dataset override. Defaults to the dataset in config.yaml beside the checkpoint.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Directory for attack_metrics.npz (default: checkpoint directory).'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='Overwrite cached attack metrics instead of skipping computation.'
    )
    append_directory_clargs(parser)
    args = parser.parse_args()
    init_directories(vars(args), load_directory_config())

    dataset_id: Optional[DATASET] = args.dataset
    model_ckpt_path: Path = args.model_ckpt_path
    config_dir = model_ckpt_path.parent
    config: Optional[SupervisedTrainingConfig] = None
    if (config_dir / 'config.yaml').exists():
        with open(config_dir / 'config.yaml', 'r') as f:
            config_kw = safe_load_yaml(f)
        config = SupervisedTrainingConfig(**config_kw)
    if dataset_id is None:
        assert config is not None, 'Cannot auto-detect dataset: provide --dataset or config.yaml beside --model-ckpt-path'
        dataset_id = config.data.id
    dest: Path = args.dest or config_dir
    dest.mkdir(exist_ok=True, parents=True)
    overwrite: bool = args.overwrite
    assert isinstance(overwrite, bool)
    dataset_kwargs = {
        'target_byte': config.data.target_byte,
        'target_variable': config.data.target_variable,
    } if config is not None else {}

    dest_path = dest / 'attack_metrics.npz'
    identity = artifact_identity(model_ckpt_path, dataset_id, dict(
        targets=dataset_kwargs, partition='attack', cropped_traces=False,
        precision='bf16-mixed',
        attack_order_seeds='0..attack_count-1', preprocessing='checkpoint config',
    ))
    if cache_matches(dest_path, identity, overwrite):
        logging.info(f'File `{dest_path}` already exists. Skipping computation.')
    else:
        attack_metrics = run_attack_performance_evalutaion(model_ckpt_path, dataset_id, dataset_kwargs)
        a, t, b = attack_metrics['ranks_zero_based'].shape
        metadata = dict(identity=identity, attack_count=a, horizon=t,
                        ranks_zero_based='uint8 [A,T,B], evaluator one-based rank minus 1',
                        guessing_entropy='one-based mean rank [B,T]',
                        success_rate='fraction rank == 1 [B,T]',
                        full_key_success_rate='fraction all outputs simultaneously rank == 1 [T]',
                        ties='pessimistic: count scores >= correct score',
                        mtd='sustained rank 1 through horizon; nonrecovery censored at horizon+1',
                        mtd_censor_sentinel=t + 1)
        np.savez_compressed(dest_path, metadata=metadata_json(metadata), **attack_metrics)
        logging.info(f'Stored attack metrics for checkpoint `{model_ckpt_path}` at `{dest_path}`.')
    with np.load(dest_path, allow_pickle=False) as metric:
        for k, v in metric.items():
            if v.ndim > 0 or k == 'metadata':
                continue
            logging.info(f'{k} = {v}')

if __name__ == '__main__':
    main()