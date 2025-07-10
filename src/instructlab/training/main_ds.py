# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import datetime
import logging
import os
import subprocess
import time
import warnings
import json
from termcolor import colored
from instructlab.training.config import Optimizer

try:
    # Third Party
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except ImportError:
    DeepSpeedCPUAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        warnings.warn(
            "DeepSpeed CPU Optimizer is not available. Some features may be unavailable.",
            UserWarning,
        )

try:
    # Third Party
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    FusedAdam = None
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if __name__ == "__main__" and (not local_rank or local_rank == 0):
        warnings.warn(
            "DeepSpeed is not available. Some features may be unavailable.",
            UserWarning,
        )

# Third Party
from tqdm import tqdm
from transformers import AutoConfig
import torch
import torch.distributed

# First Party
from instructlab.training import config
from instructlab.training.accelerator import Accelerator
from instructlab.training.config import (
    DistributedBackend,
    ModelTypes,
    TorchrunArgs,
    TrainingArgs,
)

# pylint: disable=no-name-in-module
from instructlab.training.logger import (
    propagate_package_logs,
    setup_metric_logger,
    setup_root_logger,
)
from instructlab.training.model import (
    CausalLMModel,
    LigerModel,
    Model,
    setup_optimizer,
)
from instructlab.training.multipack_sampler import (
    find_packing_max_batch_len_and_grad_accum,
)
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.utils import (
    StreamablePopen,
    check_valid_train_args,
    load_latest_full_state,
    save_checkpoint,
    save_hf_format_accelerate,
    set_random_seed,
)
import instructlab.training.data_process as dp

logger = logging.getLogger(__name__)


def train(
    args,
    model: Model,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    val_loader=None,
):
    model.train()

    # Helper function to run validation and log metrics
    def _run_validation(current_epoch: int, current_step: int):
        """Runs validation on the provided `val_loader`, aggregates the token-level
        cross entropy loss across all ranks, and logs the result.

        Args:
            current_epoch (int): The epoch we are currently on.
            current_step (int): The global **training** step we are currently on.
        """

        if val_loader is None:
            return  # nothing to do

        model.eval()

        total_loss_sum = 0.0  # accumulated sum of losses (numerator)
        total_tokens = 0.0  # accumulated number of non-padding tokens (denominator)

        # Retrieve loggers once (avoid repeated look-ups inside loop)
        metric_logger = logging.getLogger("instructlab.training.metrics")

        # Collect all validation batches first
        all_val_batches = []
        for v_batch in val_loader:
            all_val_batches.append(v_batch)

        # Find the maximum number of batches across all ranks
        # This ensures all ranks participate in the same number of forward passes
        local_batch_count = len(all_val_batches)
        max_batches_tensor = torch.tensor(
            [local_batch_count], dtype=torch.long, device=accelerator.device
        )

        # Use all_reduce to get the maximum across all ranks
        torch.distributed.all_reduce(
            max_batches_tensor, op=torch.distributed.ReduceOp.MAX
        )
        max_batches = max_batches_tensor.item()

        with torch.no_grad():
            # All ranks must go through the same number of iterations
            for i in range(max_batches):
                if i < len(all_val_batches):
                    # This rank has data for this iteration
                    v_batch = all_val_batches[i]

                    # Each batch produced by make_collate_fn contains helper metadata we need
                    num_loss_tokens = float(
                        torch.tensor([v_batch.pop("num_loss_counted_tokens")])
                    )
                    # Also pop other metadata fields that aren't tensors
                    v_batch.pop("num_samples", None)
                    v_batch.pop("total_length", None)

                    # Move tensors to the correct device
                    for k in v_batch:
                        v_batch[k] = v_batch[k].to(accelerator.device)

                    output = model(
                        **v_batch,
                        use_cache=False,
                    )

                    v_loss = output.loss
                    batch_loss = v_loss.detach().item()

                    total_loss_sum += batch_loss
                    total_tokens += num_loss_tokens
                else:
                    # This rank has no data for this iteration, but still needs to participate
                    # in collective operations. Create a dummy batch with the same structure
                    # but don't accumulate the results.
                    if len(all_val_batches) > 0:
                        # Use the structure of the first batch to create a dummy
                        dummy_batch = {}
                        first_batch = all_val_batches[0]
                        for k, v in first_batch.items():
                            if k not in [
                                "num_loss_counted_tokens",
                                "num_samples",
                                "total_length",
                            ]:
                                # Create a dummy tensor with the same shape but zeros
                                dummy_batch[k] = torch.zeros_like(v).to(
                                    accelerator.device
                                )

                        # Run forward pass with dummy data (needed for FSDP collective ops)
                        output = model(
                            **dummy_batch,
                            use_cache=False,
                        )
                        # Don't accumulate dummy results

        # Reduce the accumulated values across all ranks
        total_loss_sum, total_tokens = map(
            float,
            accelerator.reduce(
                torch.tensor(
                    [total_loss_sum, total_tokens],
                    dtype=torch.float32,
                    device=accelerator.device,
                ),
                reduction="sum",
            ),
        )

        # Compute global cross-entropy
        if total_tokens > 0:
            val_ce_loss = total_loss_sum / total_tokens
        else:
            val_ce_loss = float("nan")

        # Only log from rank 0
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            metric_logger.info(
                {
                    "epoch": current_epoch,
                    "step": current_step,
                    "rank": torch.distributed.get_rank(),
                    "validation_loss": val_ce_loss,
                },
                extra={"step": current_step},
            )

            log_data = {
                "epoch": current_epoch,
                "step": current_step,
                "rank": torch.distributed.get_rank(),
                "validation_loss": val_ce_loss,
            }
            print(colored(json.dumps(log_data, indent=2), "yellow"))

        # restore training mode
        model.train()

    global_step = 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    metric_logger = logging.getLogger("instructlab.training.metrics")
    base_logger = logging.getLogger("instructlab.training")

    batch_size = args.effective_batch_size // accelerator.grad_accum
    samples_seen = 0

    if hasattr(args, "samples_seen"):
        logger.info("Updating 'samples_seen' %d", args.samples_seen)
        samples_seen = args.samples_seen

    if accelerator.save_samples > 0:
        accelerator.save_samples = (accelerator.save_samples // batch_size) * batch_size
        logger.info("Number of samples per save: %d", args.save_samples)

    if args.save_samples_ds is not None:
        args.save_samples_ds = (args.save_samples_ds // batch_size) * batch_size
        logger.info("Number of samples per DS save: %d", args.save_samples_ds)

    global_grad_norm = None
    for epoch in range(args.current_epoch, args.num_epochs):
        if args.sampler in ("multipack"):
            accelerator.train_loader.batch_sampler.set_epoch(epoch)
        elif args.sampler in ("distributed"):
            accelerator.train_loader.sampler.set_epoch(epoch)
        else:
            raise NotADirectoryError

        num_epoch_steps = len(accelerator.train_loader)
        if local_rank == 0:
            inner_pb = tqdm(range(num_epoch_steps), desc=f"Epoch {epoch}")

        # blast through the batches in the train loader up to the last step within the epoch.
        for batch in accelerator.train_loader:
            if global_step <= args.last_step:
                # in the case of resuming, last_step > 0
                global_step += 1
                if local_rank == 0:
                    inner_pb.update(1)
                continue
            start = time.time()
            num_loss_counted_tokens = float(
                torch.tensor([batch.pop("num_loss_counted_tokens")])
            )
            micro_batch_size = float(torch.tensor([batch.pop("num_samples")]))
            total_length = float(torch.tensor([batch.pop("total_length")]))
            for k in batch:
                batch[k] = batch[k].to(local_rank)
            output = model(
                **batch,
                use_cache=False,
            )
            loss = output.loss
            log_loss = loss.detach().item()

            num_loss_counted_tokens, micro_batch_size, log_loss = map(
                float,
                accelerator.reduce(
                    torch.tensor(
                        [num_loss_counted_tokens, micro_batch_size, log_loss],
                        dtype=torch.float32,
                        device=accelerator.device,
                    ),
                    reduction="sum",
                ),
            )
            samples_seen += int(micro_batch_size)

            # num_loss_counted_tokens = aggregated_values[0]
            loss = (
                loss / num_loss_counted_tokens * world_size
            )  # dividing by the total number of non-padding tokens and multiplying by the number of GPUs so when accelerate averages by world_size, it will be the correct loss.
            base_logger.info(
                f"Epoch: {epoch}, Step: {global_step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )
            accelerator.backward(loss)

            if global_step % accelerator.grad_accum == 0:
                global_grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                accelerator.lr_scheduler.step()
                optimizer.zero_grad()

                # Run validation based on the configured schedule
                if (
                    val_loader is not None
                    and args.eval_every_n_steps > 0
                    and (global_step // accelerator.grad_accum)
                    % args.eval_every_n_steps
                    == 0
                ):
                    _run_validation(epoch, global_step)
                    rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    )
                    base_logger.info(
                        f"Rank {rank} has completed validation at step {global_step}, waiting for others..."
                    )
                    torch.distributed.barrier()

            if local_rank == 0:
                elapsed_time = time.time() - start
                overall_throughput = args.samples_per_gpu * world_size / elapsed_time
                current_lr = accelerator.lr_scheduler.get_last_lr()[0]
                cuda_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                cuda_malloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
                global_grad_norm = (
                    model.get_global_grad_norm()
                    if hasattr(model, "get_global_grad_norm")
                    else global_grad_norm
                )
                global_grad_norm = (
                    float(global_grad_norm) if global_grad_norm is not None else None
                )
                # TODO - Bring back weight_norm gather
                # weight_norm = float(
                #     model.optimizer.single_partition_of_fp32_groups[0].norm()
                # )

                # TODO - Bring back consistent gradnorm and weight_norm logging
                metric_logger.info(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "rank": torch.distributed.get_rank(),
                        "overall_throughput": overall_throughput,
                        "lr": current_lr,
                        "cuda_mem_allocated": cuda_mem_allocated,
                        "cuda_malloc_retries": cuda_malloc_retries,
                        "num_loss_counted_tokens": int(num_loss_counted_tokens),
                        "num_tokens_rank0": int(total_length),
                        "batch_size": int(micro_batch_size),
                        "total_loss": float(log_loss / num_loss_counted_tokens),
                        "samples_seen": samples_seen,
                        "gradnorm": global_grad_norm,
                        "total_samples": len(accelerator.train_loader.dataset),
                        "num_epoch_steps": num_epoch_steps,
                        # "weight_norm": weight_norm,
                    },
                    extra={"step": global_step},
                )

                log_data = {
                    "epoch": epoch,
                    "step": global_step,
                    "rank": torch.distributed.get_rank(),
                    "overall_throughput": overall_throughput,
                    "lr": current_lr,
                    "cuda_mem_allocated": cuda_mem_allocated,
                    "cuda_malloc_retries": cuda_malloc_retries,
                    "num_loss_counted_tokens": int(num_loss_counted_tokens),
                    "num_tokens_rank0": int(total_length),
                    "batch_size": int(micro_batch_size),
                    "total_loss": float(log_loss / num_loss_counted_tokens),
                    "samples_seen": samples_seen,
                    "gradnorm": global_grad_norm,
                    "total_samples": len(accelerator.train_loader.dataset),
                    "num_epoch_steps": num_epoch_steps,
                    # "weight_norm": weight_norm,
                }
                print(colored(json.dumps(log_data, indent=2), "green"))

            if args.save_samples > 0 and (
                global_step * batch_size % args.save_samples == 0
            ):
                base_logger.debug(f"Saving checkpoint at step {global_step}")
                save_checkpoint(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=model.tokenizer,
                    samples_seen=samples_seen,
                    is_lora=bool(args.lora_r),
                    hf_format=True,
                )
                base_logger.debug("RANK (%d) waiting at post-save barrier.", local_rank)
                torch.distributed.barrier()

            global_step += 1
            if local_rank == 0:
                inner_pb.update(1)
            torch.cuda.empty_cache()
        if args.checkpoint_at_epoch:
            base_logger.debug(f"Saving checkpoint at epoch {epoch}")
            save_checkpoint(
                args=args,
                accelerator=accelerator,
                model=model,
                tokenizer=model.tokenizer,
                samples_seen=samples_seen,
                is_lora=bool(args.lora_r),
                full_state=args.accelerate_full_state_at_epoch,
                hf_format=True,
                epoch=epoch,
            )
            base_logger.debug("RANK (%d) waiting at post-save barrier.", local_rank)
            torch.distributed.barrier()

    if args.save_last:
        save_hf_format_accelerate(
            args,
            model,
            model.tokenizer,
            accelerator,
            samples_seen,
            is_lora=bool(args.lora_r),
        )


# This function makes an effort to stick to a default value from torch library,
# whatever it may be. That's why we don't just set to the current (as of the
# time of writing) default: to cover the unlikely event torch decides to tweak
# the default.
def _get_collective_timeout() -> datetime.timedelta | None:
    timeout_var = os.getenv("INSTRUCTLAB_NCCL_TIMEOUT_MS")
    if timeout_var is None:
        return None

    try:
        timeout = int(timeout_var)
    except ValueError:
        timeout = -1

    if timeout <= 0:
        raise ValueError(
            f"Invalid value for INSTRUCTLAB_NCCL_TIMEOUT_MS: {timeout_var}. Must be a positive integer."
        )

    return datetime.timedelta(milliseconds=timeout)


def main(args):
    if args.distributed_training_framework == "deepspeed" and not FusedAdam:
        raise ImportError(
            "DeepSpeed was selected but we cannot import the `FusedAdam` optimizer"
        )

    if (
        args.distributed_training_framework == "deepspeed"
        and args.cpu_offload_optimizer
        and not DeepSpeedCPUAdam
    ):
        raise ImportError(
            "DeepSpeed was selected and CPU offloading was requested, but DeepSpeedCPUAdam could not be imported. This likely means you need to build DeepSpeed with the CPU adam flags."
        )

    setup_metric_logger(args.logger_type, args.run_name, args.output_dir)
    metric_logger = logging.getLogger("instructlab.training.metrics")
    if os.environ["LOCAL_RANK"] == "0":
        metric_logger.info(vars(args), extra={"hparams": True})

    setup_root_logger(args.log_level)
    tokenizer = setup_tokenizer(args.model_name_or_path, args.chat_tmpl_path)
    # device = torch.device("cuda", args.local_rank)

    model_conf = AutoConfig.from_pretrained(args.model_name_or_path)
    args.model_type = model_conf.model_type

    #### distributed init #####
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])

    timeout = _get_collective_timeout()
    if timeout is not None:
        torch.distributed.init_process_group(timeout=timeout)
    else:
        torch.distributed.init_process_group()

    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    flash_enabled = Model.check_flash_attn_enabled(args.disable_flash_attn)

    # ---------------------------------------------------------------------
    # Dataset preparation (train / validation split)
    # ---------------------------------------------------------------------

    full_dataset = setup_dataset(
        args.data_path,
        mock=args.mock_data,
        mock_len=args.mock_len,
    )

    # Perform a deterministic split so that each rank sees the same partition.
    # When `validation_split` is 0 or < 0, we skip creating a validation set.
    from torch.utils.data import (
        random_split,
        Subset,
    )  # local import to avoid polluting top of file

    if args.validation_split and args.validation_split > 0.0:
        train_len = int(len(full_dataset) * (1 - args.validation_split))
        val_len = len(full_dataset) - train_len

        # Ensure positive lengths
        if train_len == 0 or val_len == 0:
            raise ValueError(
                "`validation_split` resulted in an empty train or validation split. Please adjust the value."
            )

        generator = torch.Generator().manual_seed(args.seed)

        train_subset, val_subset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )

        class _SubsetWithLengths(Subset):
            """torch.utils.data.Subset that retains the `get_lengths` method used by Multipack."""

            def get_lengths(self):  # type: ignore[override]
                import numpy as _np

                base_lengths = self.dataset.get_lengths()
                # Convert indices to numpy array for advanced indexing
                return _np.asarray(base_lengths)[self.indices]

        train_dataset = _SubsetWithLengths(full_dataset, train_subset.indices)
        val_dataset = _SubsetWithLengths(full_dataset, val_subset.indices)
    else:
        train_dataset = full_dataset
        val_dataset = None

    # This model class wraps the various AutoModel classes we support
    # based on model_type, and model_path -> choose auto_model
    lora_config = None

    if args.lora_r > 0:
        lora_config = Model.create_lora_config(
            lora_target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_r=args.lora_r,
        )

    # Create model based on type
    model_class_map = {
        ModelTypes.LIGER: LigerModel,
        ModelTypes.CAUSALLM: CausalLMModel,
    }

    # Convert string to ModelTypes enum with fallback
    try:
        model_type = ModelTypes(args.model_class)
    except (ValueError, AttributeError):
        model_type = ModelTypes.CAUSALLM

    # Get the model class with default fallback
    model_class = model_class_map.get(model_type, CausalLMModel)
    m = model_class(
        model_path=args.model_name_or_path,
        output_dir=args.output_dir,
        lora_config=lora_config,
        distributed_framework=DistributedBackend(args.distributed_training_framework),
        tokenizer=tokenizer,
        flash_enabled=flash_enabled,
        noise_alpha=args.NEFTune_alpha,
        lora_quant_bits=args.lora_quant_bits,
    )

    args.base_model_args = m.base_model_args

    try:
        packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
            num_gpus=torch.distributed.get_world_size(),
            avg_sample_len=train_dataset.get_lengths().mean(),
            effective_batch_size=args.effective_batch_size,
            max_batch_len_per_gpu=args.max_batch_len,
            is_padding=not flash_enabled,
            dataset=train_dataset,
            seed=args.seed,
        )
        args.sampler = "multipack"
    except RuntimeError as e:
        logger.error(e)

        # fallback to grad accum = 1
        # NOTE: packing max batch len will not be used
        packing_max_batch_len = None
        grad_accum = 1
        args.sampler = "distributed"

    args.samples_per_gpu = (
        args.effective_batch_size // grad_accum // torch.distributed.get_world_size()
    )

    # Validation DataLoader (if applicable)
    val_loader = None
    if val_dataset is not None:
        val_loader = setup_dataloader(
            val_dataset,
            tokenizer.pad_token_id,
            num_workers=4,
            flash_enabled=flash_enabled,
            max_batch_len=args.max_batch_len,
            packing_max_batch_len=packing_max_batch_len,
            samples_per_gpu=args.samples_per_gpu,
            sampler="distributed",  # simpler eval sampling
            seed=args.seed,
        )
        logger.info(
            f"Rank {torch.distributed.get_rank()}: Created validation loader with {len(val_loader)} batches from {len(val_dataset)} samples"
        )

    train_loader = setup_dataloader(
        train_dataset,
        tokenizer.pad_token_id,
        num_workers=8,
        flash_enabled=flash_enabled,
        max_batch_len=args.max_batch_len,
        packing_max_batch_len=packing_max_batch_len,
        samples_per_gpu=args.samples_per_gpu,
        sampler=args.sampler,
        seed=args.seed,
    )
    if len(train_loader) == 0:
        # this happens sometimes when we have more GPUs than data to process. In this case
        # we should either alert the user to switch samplers, or do it automatically and
        # warn them about it happening
        logger.warning(
            "The dataset is too small for multipack to distribute all of the samples across GPUs. Falling back to the distributed sampler!"
        )
        args.sampler = "distributed"
        train_loader = setup_dataloader(
            train_dataset,
            tokenizer.pad_token_id,
            num_workers=8,
            flash_enabled=flash_enabled,
            max_batch_len=args.max_batch_len,
            packing_max_batch_len=packing_max_batch_len,
            samples_per_gpu=args.samples_per_gpu,
            sampler=args.sampler,
            seed=args.seed,
        )

    if args.local_rank == 0:
        metric_logger.info(
            {
                "num_gpus": torch.distributed.get_world_size(),
                "avg_sample_len": train_dataset.get_lengths().mean(),
                "effective_batch_size": args.effective_batch_size,
                "max_batch_len_per_gpu": args.max_batch_len,
                "packing_max_batch_len": packing_max_batch_len,
                "grad_accum": grad_accum,
                "num_batches": len(train_loader),
                "avg_samples_per_batch": len(train_dataset) / len(train_loader),
                "samples_per_gpu": args.samples_per_gpu,
                "total_samples": len(train_dataset),  # emit the total number of samples
            },
            extra={"hparams": True},
        )
    # accelerator does not need optimizer to init, in fact, the optimizer needs to be initialized AFTER the Accelerator
    accelerator = Accelerator(
        model=m,
        samples_per_gpu=args.samples_per_gpu,
        grad_accum=grad_accum,
        train_loader=train_loader,
        distributed_framework=DistributedBackend(args.distributed_training_framework),
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        deepspeed_cpu_offload_optimizer=args.cpu_offload_optimizer,
        deepspeed_cpu_offload_optimizer_pin_memory=args.cpu_offload_optimizer_pin_memory,
        deepspeed_cpu_offload_optimizer_ratio=args.cpu_offload_optimizer_ratio,
        fsdp_cpu_offload_params=args.cpu_offload_params_fsdp,
        save_samples=args.save_samples,
    )
    # optimizer needs model that has been prepared by accelerator
    # and then accelerator needs to be prepared AGAIN once optimizer is initialized
    optimizer = setup_optimizer(
        model=m,
        cpu_offload=args.cpu_offload_optimizer,
        name=args.optimizer,  # choose based on backend
        learning_rate=args.learning_rate,
    )
    accelerator.prepare_with_optimizer(
        optimizer=optimizer,
        lr_scheduler=args.lr_scheduler,
        num_epochs=args.num_epochs,
        num_warmup_steps=args.num_warmup_steps,
    )
    # TODO: make this work more seamlessly
    optimizer = accelerator.optimizer
    m = accelerator.model

    # Ensure the validation DataLoader is wrapped by Accelerate for proper
    # device placement and distributed handling.
    if val_loader is not None:
        from copy import deepcopy as _deepcopy

        val_loader = accelerator.accelerator.prepare(_deepcopy(val_loader))

    load_latest_full_state(args=args, accelerator=accelerator)

    train(
        args,
        model=m,
        optimizer=optimizer,
        accelerator=accelerator,
        val_loader=val_loader,
    )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


# public API
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    """
    # Set up logging first before any processing
    # Enable package logging propagation before setting up loggers
    propagate_package_logs(True)
    setup_root_logger(train_args.log_level)
    setup_metric_logger(
        train_args.logger_type, train_args.run_name, train_args.ckpt_output_dir
    )

    logger = logging.getLogger("instructlab.training")
    logger.info("Starting training setup...")

    check_valid_train_args(train_args)

    # switch out generic tmpl for legacy tmpl if requested
    if train_args.use_legacy_tmpl:
        train_args.chat_tmpl_path = os.path.join(
            os.path.dirname(__file__), "chat_templates/ibm_legacy_tmpl.py"
        )

    if train_args.process_data:
        # TODO(osilkin):
        #   Decouple the data processing logic from training.
        #   Now that we've decided that repos will be less tethered to the
        #   design choices of the `ilab` CLI, we can make this change.
        dp.process_data(
            data_output_path=train_args.data_output_dir,
            model_path=train_args.model_path,
            data_path=train_args.data_path,
            max_seq_len=train_args.max_seq_len,
            chat_tmpl_path=train_args.chat_tmpl_path,
            num_cpu_procs=train_args.data_process_num_cpu_procs,
        )

    if not os.path.exists(train_args.ckpt_output_dir):
        os.makedirs(train_args.ckpt_output_dir, exist_ok=True)

    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node_rank={torch_args.node_rank}",
        f"--nproc_per_node={torch_args.nproc_per_node}",
        f"--rdzv_id={torch_args.rdzv_id}",
        f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
        __file__,
        f"--model_name_or_path={train_args.model_path}",
        f"--data_path={train_args.data_output_dir}/data.jsonl",
        f"--output_dir={train_args.ckpt_output_dir}",
        f"--num_epochs={train_args.num_epochs}",
        f"--effective_batch_size={train_args.effective_batch_size}",
        f"--learning_rate={train_args.learning_rate}",
        f"--num_warmup_steps={train_args.warmup_steps}",
        f"--save_samples={train_args.save_samples}",
        f"--log_level={train_args.log_level}",
        f"--max_batch_len={train_args.max_batch_len}",
        f"--seed={train_args.random_seed}",
        f"--logger_type={train_args.logger_type}",
        f"--eval_every_n_steps={train_args.eval_every_n_steps}",
        f"--validation_split={train_args.validation_split}",
        f"--optimizer={train_args.optimizer.value}",
    ]

    if train_args.chat_tmpl_path is not None:
        command.append(f"--chat-tmpl-path={train_args.chat_tmpl_path}")

    if train_args.run_name is not None:
        command.append(f"--run_name={train_args.run_name}")

    if train_args.use_liger:
        command.append("--use_liger")

    if train_args.keep_last_checkpoint_only:
        command.append("--keep_last_checkpoint_only")

    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint_at_epoch")

    if train_args.accelerate_full_state_at_epoch:
        command.append("--accelerate_full_state_at_epoch")

    if train_args.mock_data:
        command.append("--mock_data")
        if train_args.mock_len:
            command.append(f"--mock_len={train_args.mock_len}")

    if train_args.disable_flash_attn:
        command.append("--disable_flash_attn")

    if train_args.lora:
        command.extend(
            [
                f"--lora_r={train_args.lora.rank}",
                f"--lora_alpha={train_args.lora.alpha}",
                f"--lora_dropout={train_args.lora.dropout}",
                "--lora_target_modules",
            ]
        )
        if train_args.lora.target_modules:
            command.extend(train_args.lora.target_modules)
        # hard-code 4-bit quantization for now, change this when we add more
        quant_dtype = train_args.lora.quantize_data_type
        quantization_is_enabled = quant_dtype in (
            config.QuantizeDataType.NF4,
            config.QuantizeDataType.NF4.value,
        )
        if quantization_is_enabled:
            command.append("--lora_quant_bits=4")

    # specify which distributed training backend we use
    command.append(
        f"--distributed_training_framework={train_args.distributed_backend.value}"
    )

    # deepspeed options
    if train_args.distributed_backend == DistributedBackend.DEEPSPEED:
        if not FusedAdam:
            raise ImportError(
                "DeepSpeed was selected as the distributed backend, but FusedAdam could not be imported. Please double-check that DeepSpeed is installed correctly"
            )

        if train_args.deepspeed_options.cpu_offload_optimizer and not DeepSpeedCPUAdam:
            raise ImportError(
                "DeepSpeed CPU offloading was enabled, but DeepSpeedCPUAdam could not be imported. This is most likely because DeepSpeed was not built with CPU Adam. Please rebuild DeepSpeed to have CPU Adam, or disable CPU offloading."
            )
    if train_args.deepspeed_options.save_samples:
        command.append(f"--save_samples_ds={train_args.deepspeed_options.save_samples}")
    if train_args.deepspeed_options.cpu_offload_optimizer:
        command.extend(
            [
                "--cpu_offload_optimizer",
                f"--cpu_offload_optimizer_ratio={train_args.deepspeed_options.cpu_offload_optimizer_ratio}",
            ]
        )
        if train_args.deepspeed_options.cpu_offload_optimizer_pin_memory:
            command.append("--cpu_offload_optimizer_pin_memory")

    # FSDP Options
    if train_args.fsdp_options.cpu_offload_params:
        command.extend(
            [
                "--cpu_offload_params_fsdp",
            ]
        )

    # specify the sharding strategy
    command.append(
        f"--fsdp_sharding_strategy={train_args.fsdp_options.sharding_strategy.value}"
    )

    if train_args.keep_last_checkpoint_only:
        command.append("--keep_last_checkpoint_only")

    logger.info("Running training command as subprocess: %s", " ".join(command))
    process = None
    interrupt: KeyboardInterrupt | Exception | None = None
    failure = False
    try:
        process = StreamablePopen(
            f"{train_args.ckpt_output_dir}/full_logs_global{torch_args.node_rank}.log",
            command,
        )
        print(" ".join(command))
        process.listen()
    except KeyboardInterrupt as e:
        logger.info("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        logger.error(
            "Unexpected exception received during distributed training", exc_info=e
        )
        interrupt = e
    finally:
        if "process" not in locals() or process is None:
            return

        failure = process.poll() != 0
        if not failure:
            logger.info("Operation completed successfully! 🎉")
        else:
            logger.error("Training subprocess has not exited yet. Sending SIGTERM.")

        process.terminate()
        try:
            logger.info("Waiting for process to exit, 60s...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.error(
                "Training subprocess did not terminate before timeout, sending SIGKILL."
            )
            process.kill()

        if interrupt:
            raise interrupt
        if failure:
            raise RuntimeError(
                "Suffered a failure during distributed training. Please see the training logs for more context."
            )


if __name__ == "__main__":
    # TODO(osilkin): Configure a type that these args must adhere to for the sake of type checking
    #               Maybe switch out from argparse to something smarter
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument(
        "--model-class",
        type=str,
        default=ModelTypes.CAUSALLM.value,
        help=f"valid model classes are {[x.value for x in ModelTypes]}.",
    )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--current_epoch",
        type=int,
        default=0,
        help="Helpful flag for resuming on a later epoch. Sets dataloader correctly.",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=0,
        help="understand this as the last completed step. "
        "The default is 0, since global_step starts from 1 by default.",
    )
    # parser.add_argument("--samples_per_gpu", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=3840)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--save_samples",
        type=int,
        help="The number of samples seen between each checkpoint save. If --save_samples<=0, this feature is disabled.",
    )
    parser.add_argument(
        "--save_samples_ds",
        type=int,
        help="for saving in ds native format",
        default=None,
    )
    parser.add_argument(
        "--save_last", action="store_true", help="save after finishing training"
    )
    parser.add_argument(
        "--checkpoint_at_epoch",
        action="store_true",
        help="Save a model checkpoint after finishing an epoch.",
    )
    parser.add_argument(
        "--accelerate_full_state_at_epoch",
        action="store_true",
        help="Save full model state using Accelerate after finishing an epoch.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logger_type", type=str, default="async")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock_data", action="store_true")
    parser.add_argument("--mock_len", type=int, default=2600)
    parser.add_argument(
        "--distributed_training_framework",
        type=str,
        choices=[
            DistributedBackend.DEEPSPEED.value,
            DistributedBackend.FSDP.value,
        ],
        default=DistributedBackend.DEEPSPEED.value,
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        # choices=[e.name for e in ShardingStrategy],
        default="HYBRID_SHARD",
        help="Sharding strategy to be used for FSDP distributed training.",
    )
    parser.add_argument(
        "--use_dolomite",
        action="store_true",
        help="(Deprecated, NoOp) Attempts to use GPTDolomite architecture",
    )
    parser.add_argument("--lora_r", type=int, default=0)  # set to > 0 to activate lora
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_quant_bits", type=int, default=None)
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=None,
        help="Which modules we should target for injecting LoRA layers. Defaults to selecting all projection layers when no values are provided.",
    )
    parser.add_argument("--max_batch_len", type=int, default=60000)
    parser.add_argument(
        "--cpu_offload_optimizer",
        action="store_true",
        default=False,
        help="Offload optimizer to CPU when using DeepSpeed. This configures it to use ZeRO stage 2.",
    )
    parser.add_argument(
        "--cpu_offload_params_fsdp",
        action="store_true",
        default=False,
        help="Offload to CPU when using FSDP.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_pin_memory",
        action="store_true",
        default=False,
        help="Pin memory when offloading optimizer to CPU. This allows for faster transfers between CPU and GPU. Comes at the cost of higher memory usage and CPU overhead.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_ratio",
        type=float,
        default=1.0,
        help="Ratio of the optimizer to be offloaded to CPU. The rest will be on GPU(s).",
    )
    parser.add_argument("--NEFTune_alpha", type=float, default=None)
    parser.add_argument(
        # TODO(osilkin): rename to chat_tmpl_path
        "--chat-tmpl-path",
        type=str,
        default=None,
        help="Path to the chat template to set on the model for training. If none is provided, the chat template used in the model will be used.",
    )
    parser.add_argument("--disable_flash_attn", action="store_true")
    parser.add_argument(
        "--keep_last_checkpoint_only",
        action="store_true",
        help=(
            "Keep only the last checkpoint directory - overwrite the previous ones. Useful for saving disk space."
            "The last checkpoint will be saved as 'last_epoch'."
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=Optimizer.ADAMW.value,
        help="The optimizer to use.",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=100,
        help="How often to evaluate the model. This is the number of steps (calls to `optimizer.step()`) between evaluations.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="The fraction of the training data to use for validation. This is the ratio of the training data to the validation data.",
    )
    parser.add_argument(
        "--use_liger",
        action="store_true",
        help="Use Liger kernels for training.",
    )
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)

"""
pkill python
git reset --hard
git pull
export WORLD_SIZE=1
sleep 3
mkdir -p /new_data/experiments/ap-fsdp-p00-old-m-ds-2t
cd /app/fsdp
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=mistralai/Mistral-7B-v0.1 \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-fsdp-p00-old-m-ds-2t" \
--num_epochs=100 \
--samples_per_gpu=24 \
--learning_rate=1e-06 \
--num_warmup_steps=800 \
--gradient_accumulation_steps=2 \
--save_samples=12000 \
--log_level="INFO" \
--mock_data \
--mock_len=2048 \
--seed=42 | tee /new_data/experiments/ap-fsdp-p00-old-m-ds-2t/$RANK.log
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=/new_data/models/granite7b/ibm_models_version/ \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-granite-4t" \
--num_epochs=100 \
--samples_per_gpu=240 \
--learning_rate=2e-05 \
--num_warmup_steps=385 \
--gradient_accumulation_steps=2 \
--save_samples=250000 \
--log_level="INFO" \
--fsdp_sharding_strategy="SHARD_GRAD_OP" \
--max_batch_len 70000 \
--seed=42
"""
