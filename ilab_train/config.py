from dataclasses import dataclass
import yaml

@dataclass
class TorchrunTrainArgs:
    """
    Representation of the arguments being used by torchrun.
    The full list of arguments can be found here:
    https://pytorch.org/docs/stable/elastic/run.html#definitions
    """
    nproc_per_node: int
    nnodes: int
    node_rank: int
    rdzv_id: int
    rdzv_endpoint: str

    def __str__(self):
        return yaml.dump(vars(self), sort_keys=False)


@dataclass
class FullTrainArgs:
    """
    This class represents the arguments being used by the training script.
    """
    data_path: str
    input_dir: str
    model_name_or_path: str
    output_dir: str
    num_epochs: int
    effective_batch_size: int
    learning_rate: float
    num_warmup_steps: int
    save_samples: int
    log_level: str
    seed: int
    mock_data: bool
    mock_len: int
    is_granite: bool
    max_batch_len: int
    # I don't believe this is actually used anywhere anymore,
    # but we should still keep it to avoid changing too much at once
    samples_per_gpu: int

    def __str__(self):
        return yaml.dump(vars(self), sort_keys=False)

