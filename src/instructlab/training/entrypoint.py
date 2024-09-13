# SPDX-License-Identifier: Apache-2.0
from instructlab.training.config import (
    TorchrunArgs,
    TrainingArgs,
    DistributedTrainingBackend,
)

"""
This file is intended to contain the primary entrypoint for when users call out to run training
"""


# public API
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Primary entrypoint for training LLM models with the InstructLab training library.

    Note: this training library supports both FSDP and DeepSpeed as distributed training frameworks.
    Please read the documentation on `train_args` for a further explanation of how they can be used.

    Args:
        torch_args (`TorchrunArgs`):
            Specifies how torchrun should orchestrate multi-GPU training. For single-GPU instances,
            you can simply set `torch_args.nproc_per_node` to 1.

        train_args (`TrainingArgs`):
            This specifies parameters of the training job such as the learning rate, effective batch size,
            number of epochs to run for, etc.

            Note: The default distributed training framework is DeepSpeed. To change this behavior, set
            `distributed_training_backend = "fsdp"` to use FSDP instead.

    """
    match train_args.distributed_training_backend:
        case DistributedTrainingBackend.DEEPSPEED:
            from .main_ds import run_training as _run_training

            _run_training(torch_args=torch_args, train_args=train_args)
        case DistributedTrainingBackend.FSDP:
            from .main_fsdp import run_training as _run_training

            _run_training(torch_args=torch_args, train_args=train_args)
        case _:
            raise ValueError(
                f"Unsupported distributed training backend: '{train_args.distributed_training_backend}'"
            )
