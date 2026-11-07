"""Adapt RHSCI samples to the native GR00T shard partition and mixture.

The adapter only translates episode slots into the existing RGB/ARDY reader.
Partitioning, shard weighting/cycling, rank/worker filtering, statistics merging
and processing order remain in the native dataset implementations.
"""

import numpy as np
import torch.distributed as dist

from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import MessageType

from .data import TAG, modality, statistics


class EpisodeView:
    """Native loader interface over already-validated, dataset-local episodes."""

    def __init__(self, samples):
        self.samples = samples
        self.episode_lengths = [episode["length"] for episode in samples.episodes]

    def get_episode_length(self, slot):
        return self.episode_lengths[slot]

    def __getitem__(self, slot):
        return slot

    def get_dataset_statistics(self):
        return statistics(self.samples.root, self.samples.variant)[TAG.value]


class RHSCIShardedDataset(ShardedSingleStepDataset):
    """Reuse native partition/get_shard; retain local language and ARDY identity."""

    def __init__(self, samples, *, shard_size, episode_sampling_rate, seed):
        ShardedDataset.__init__(self, samples.root)
        self.samples = samples
        self.embodiment_tag = TAG
        self.modality_configs = modality(samples.variant)[TAG.value]
        self.action_horizon = 16
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.processor = None
        self.episode_loader = EpisodeView(samples)
        self.shard_dataset()

    def get_datapoint(self, episode_slot, step_index):
        start = int(self.samples.ends[episode_slot - 1]) if episode_slot else 0
        sample = self.samples.raw_sample(start + int(step_index))
        # Native get_shard processes in shard order; do not use the legacy
        # map reader's per-(epoch,index) RNG reseeding around augmentation.
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": sample}])


class RHSCIShardedMixture(ShardedMixtureDataset):
    """Native iterator with explicit lifecycle cleanup for its prefetch thread."""

    def bind_distributed(self):
        # Trainer/Accelerate may initialize DDP after dataset construction.
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def __iter__(self):
        try:
            yield from super().__iter__()
        finally:
            self.close()

    def close(self):
        if self._cache_job is not None:
            self._cache_job.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = self._cache_job = self.curr_shard = None


def build_sharded_mixture(paths, dataset, processor):
    """One root per native dataset group; CLI weights are group mix_ratios."""
    from rhsci.data.multitask import shared_statistics

    config = paths["shard_config"]
    shards = [
        RHSCIShardedDataset(
            child,
            **{key: config[key] for key in ("shard_size", "episode_sampling_rate", "seed")},
        )
        for child in dataset.datasets
    ]
    mixture = RHSCIShardedMixture(
        shards,
        paths["dataset_weights"],
        processor,
        seed=config["seed"],
        num_shards_per_epoch=config["num_shards_per_epoch"],
        override_pretraining_statistics=False,
    )
    expected = statistics(
        paths["dataset"], dataset.datasets[0].variant, raw=shared_statistics(paths, "gr00t")
    )[TAG.value]
    if mixture.global_stats[TAG.value] != expected:
        raise ValueError("Native mixture statistics differ from the saved joint contract")
    return mixture
