from dataclasses import dataclass
import yaml
from enum import Enum


class DeepSpeedOffloadStrategy(Enum):
    """
    Defines the offload strategy for DeepSpeed.

    To learn more, read about it here: https://www.deepspeed.ai/tutorials/zero-offload/
    """

    CPU = "cpu"
    NVME = "nvme"
    NONE = None


class QuantizeDataType(Enum):
    """
    Defines what datatype we use during quantization.
    """

    NF4 = "nf4"
    FP8 = "fp8"
    NONE = None


class YAMLAble:
    """
    For our classes to easily be printable
    """

    def __str__(self):
        return yaml.dump(vars(self), sort_keys=False)


@dataclass
class DataProcessArgs(YAMLAble):
    """
    All the arguments consumed by the training data pre-process script.
    """

    data_path: str
    data_output_path: str
    max_seq_len: str  # defines the max sequence length of a sample
    model_path: str  # either a HF model name or path to HF model


@dataclass
class TorchrunTrainArgs(YAMLAble):
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


@dataclass
class LoraOptions:
    """
    Options to specify when training using a LoRA.
    """

    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: list


@dataclass
class FullTrainArgs(YAMLAble):
    """
    This class represents the arguments being used by the training script.
    """

    # Either the name of a HuggingFace model or a path to a model saved in HuggingFace format.
    model_path: str

    # this field specifies the filepath to the training dataset before processing
    data_path: str
    ckpt_output_path: str

    # this field defines where we should be saving the processed version of the training dataset
    # after we have tokenized it
    processed_data_output_path: str

    num_gpus: int
    max_seq_len: int
    max_batch_len: int
    num_epochs: int
    effective_batch_size: int
    save_samples: int
    learning_rate: float
    warmup_steps: int

    ds_offload_strat: DeepSpeedOffloadStrategy
    cpu_offload_optimizer: bool
    cpu_offload_params: bool

    quantize_dtype: QuantizeDataType
    lora: LoraOptions | None
