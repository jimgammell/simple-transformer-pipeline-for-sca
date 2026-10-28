from typing import List, get_args
import argparse

from torch.utils.data import DataLoader
from uncropped_transformers.training.supervised_lightning_module import SupervisedModule
from uncropped_transformers.datasets import Base_TorchDataset
from uncropped_transformers.deep_attribution.attributor import Attributor, ATTRIBUTION_METHOD

from init_things import *
from utils.training_config import SupervisedTrainingConfig
from utils.load_things import load_torch_dataset, construct_loaders, load_trained_model
from utils.artifacts import artifact_identity, metadata_json, cache_matches

torch.backends.cudnn.benchmark = False
 
def compute_feature_attribution(
        module: SupervisedModule,
        profiling_loader: DataLoader,
        dest_dir: Path,
        methods: List[ATTRIBUTION_METHOD],
        identities: dict,
        overwrite: bool = False,
):
    for method in methods:
        dest = dest_dir / f'{dash_to_uscr(method)}.npz'
        if cache_matches(dest, identities[method], overwrite):
            continue
        module.to('cuda')
        module.eval()
        attributor = Attributor(module)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            attr = attributor(method, profiling_loader, show_progress_bar=True).float().cpu().numpy()
            np.savez_compressed(dest, attributions=attr,
                                metadata=metadata_json(dict(identity=identities[method])))

def main():
    parser = argparse.ArgumentParser(description='Compute selected feature attributions from a checkpoint using profiling traces.')
    parser.add_argument('--ckpt-path', required=True, type=Path)
    parser.add_argument('--config-path', type=Path, default=None)
    parser.add_argument('--dest', type=Path, default=None)
    parser.add_argument('--attr-methods', type=str, default=[], nargs='*')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    append_directory_clargs(parser)
    args = parser.parse_args()
    init_directories(vars(args), load_directory_config())

    ckpt_path: Path = args.ckpt_path
    assert ckpt_path.exists()
    config_path: Optional[Path] = args.config_path
    if config_path is None:
        config_path = ckpt_path.parent / 'config.yaml'
    assert config_path.exists()
    with open(config_path, 'r') as f:
        config_kw = safe_load_yaml(f)
    config = SupervisedTrainingConfig(**config_kw)
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = ckpt_path.parent
    dest.mkdir(exist_ok=True, parents=True)
    attr_methods: List[ATTRIBUTION_METHOD] = args.attr_methods
    assert all(x in get_args(ATTRIBUTION_METHOD) for x in attr_methods)
    batch_size: Optional[int] = args.batch_size
    if batch_size is not None:
        assert isinstance(batch_size, int) and batch_size > 0
        config.training.batch_size = batch_size
    
    dataset_kwargs = {
        'target_byte': config.data.target_byte,
        'target_variable': config.data.target_variable,
    }
    identities = {method: artifact_identity(ckpt_path, config.data.id, dict(
        targets=dataset_kwargs, partition='profile', cropped_traces=False,
        preprocessing=config.data.preprocessing, precision='bf16-mixed',
        batch_size=config.training.batch_size, method=method,
        smoothing_std=0.0, smoothing_count=1, aggregation='mean absolute per-example attribution',
    )) for method in attr_methods}
    pending: List[ATTRIBUTION_METHOD] = [method for method in attr_methods if not cache_matches(
        dest / f'{dash_to_uscr(method)}.npz', identities[method], args.overwrite)]
    if not pending:
        return
    profiling_set = load_torch_dataset(config.data.id, 'profile', **dataset_kwargs)
    profiling_loader, = construct_loaders([], [profiling_set], batch_size=config.training.batch_size)
    module = load_trained_model(ckpt_path, profiling_set)
    compute_feature_attribution(module, profiling_loader, dest, pending, identities, args.overwrite)

if __name__ == '__main__':
    main()