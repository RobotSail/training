from dataclasses import dataclass, field
import yaml
from enum import Enum
from omegaconf import MISSING


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
    # FP8 = "fp8" TODO: test and evaluate fp8
    NONE = None


@dataclass
class DataProcessArgs:
    """
    All the arguments consumed by the training data pre-process script.
    """

    data_path: str
    data_output_path: str
    max_seq_len: str  # defines the max sequence length of a sample
    model_path: str  # either a HF model name or path to HF model


@dataclass
class TorchrunArgs:
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

    rank: int = 4
    alpha: float = 32
    dropout: float = 0.1
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DeepSpeedOptions:
    """
    Represents the available options we support when training with the DeepSpeed optimizer.
    For more information, please read:
    https://www.deepspeed.ai/docs/config-json/
    """

    ds_offload_strat: DeepSpeedOffloadStrategy = MISSING
    cpu_offload_optimizer: bool = MISSING


@dataclass
class FullTrainArgs:
    """
    This class represents the arguments being used by the training script.
    """

    # Either the name of a HuggingFace model or a path to a model saved in HuggingFace format.
    model_path: str = MISSING

    # this field specifies the filepath to the training dataset before processing
    data_path: str = MISSING
    ckpt_output_path: str = MISSING

    # this field defines where we should be saving the processed version of the training dataset
    # after we have tokenized it
    data_output_dir: str = MISSING

    max_seq_len: int = MISSING
    max_batch_len: int = MISSING
    num_epochs: int = MISSING
    effective_batch_size: int = MISSING
    save_samples: int = MISSING
    learning_rate: float = MISSING
    warmup_steps: int = MISSING
    is_padding_free: bool = MISSING
    random_seed: int = MISSING

    mock_data: bool = False
    mock_data_len: int = 0

    deepspeed_options: DeepSpeedOptions = field(
        default_factory=lambda: DeepSpeedOptions(
            ds_offload_strat=DeepSpeedOffloadStrategy.NONE, cpu_offload_optimizer=False
        )
    )

    quantize_dtype: QuantizeDataType = QuantizeDataType.NONE
    lora: LoraOptions | None = None
