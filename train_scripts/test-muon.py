from instructlab.training import run_training, TrainingArgs, TorchrunArgs
import datasets
import os


from instructlab.training import (
    run_training,
    TrainingArgs,
    TorchrunArgs,
    FSDPOptions,
    ShardingStrategies,
)
import datasets


def granite_31_8b_v2_bmo():
    # setup the torchrun args here
    torchrun_args = TorchrunArgs(
        nproc_per_node=8,
        nnodes=1,
        node_rank=0,
        rdzv_id=420,
        rdzv_endpoint="0.0.0.0:12345",
    )

    # first train the model with p07
    p07_data_path = "/mnt/7TB-a/datasets/ad-bmo-generated-04172025/knowledge_train_msgs_2025-04-17T19_54_20.jsonl"
    # p07_ckpt_output_dir = "/mnt/7TB-a/models/granite-3.1-8b-v2-BMO-p07_muon-1e4"
    p07_ckpt_output_dir = "/mnt/7TB-a/models/granite-3.1-8b-v2-BMO-p07_adamw-baseline"
    p07_input_model = "/mnt/7TB-a/models/granite-3.1-8b-starter-v2"
    max_batch_len = 45_000
    max_seq_len = 40_000

    # p10_data_path = "/mnt/7TB-a/datasets/ad-bmo-generated-04172025/skills_train_msgs_2025-04-17T19_54_20.jsonl"
    # p10_ckpt_output_dir = "/mnt/nvme1n1/models/granite-3.1-8b-v2-BMO-p10"
    # p10_input_model = None

    # set up the p07 training args
    p07_training_args = TrainingArgs(
        data_path=p07_data_path,
        model_path=p07_input_model,
        data_output_dir="/dev/shm",
        ckpt_output_dir=p07_ckpt_output_dir,
        max_batch_len=max_batch_len,
        max_seq_len=max_seq_len,
        num_epochs=7,
        effective_batch_size=128,
        warmup_steps=10,
        save_samples=0,
        # learning_rate=1e-4,
        learning_rate=2e-5,
        checkpoint_at_epoch=True,
        accelerate_full_state_at_epoch=False,
        use_liger=False,
        distributed_backend="fsdp",
        fsdp_options=FSDPOptions(sharding_strategy=ShardingStrategies.HYBRID_SHARD),
        # Enable TensorBoard logging
        logger_type="tensorboard",  # or "async,tensorboard" for both
        run_name="my_experiment_{time}_adamw",  # Will include timestamp
        optimizer="Adamw",
    )
    run_training(torchrun_args, p07_training_args)

    # set up the p10 training args
    p10_input_model = None
    p07_checkpoint_location = f"{p07_ckpt_output_dir}/hf_format"
    p07_checkpoints = os.listdir(p07_checkpoint_location)
    most_recent_checkpoint, most_recent_time = None, 0
    for checkpoint in p07_checkpoints:
        full_ckpt_path = f"{p07_checkpoint_location}/{checkpoint}"
        if os.stat(full_ckpt_path).st_ctime > most_recent_time:
            most_recent_checkpoint = full_ckpt_path
            most_recent_time = os.stat(full_ckpt_path).st_ctime

    # print out the most recent checkpoint
    print(f"best checkpoint: {most_recent_checkpoint}")
    p10_input_model = most_recent_checkpoint
    # p10_data_path = "/mnt/7TB-a/datasets/ad-bmo-generated-04172025/skills_train_msgs_lab_v2_bmo.jsonl"
    # p10_ckpt_output_dir = "/mnt/nvme1n1/models/granite-3.1-8b-v2-BMO-p10"
    # assert p10_input_model is not None

    # # set up the p10 training args
    # p10_training_args = TrainingArgs(
    #     data_path=p10_data_path,
    #     model_path=p10_input_model,
    #     data_output_dir="/dev/shm",
    #     ckpt_output_dir=p10_ckpt_output_dir,
    #     max_batch_len=max_batch_len,
    #     max_seq_len=max_seq_len,
    #     num_epochs=7,
    #     effective_batch_size=3840,
    #     warmup_steps=25,
    #     save_samples=0,
    #     learning_rate=2e-5,
    #     checkpoint_at_epoch=True,
    #     accelerate_full_state_at_epoch=True,
    #     use_liger=False,
    #     distributed_backend="fsdp",
    #     fsdp_options=FSDPOptions(sharding_strategy=ShardingStrategies.HYBRID_SHARD),
    # )
    # run_training(torchrun_args, p10_training_args)


def granite_31_8b_v2_bmo_p10():
    # setup the torchrun args here
    torchrun_args = TorchrunArgs(
        nproc_per_node=8,
        nnodes=1,
        node_rank=0,
        rdzv_id=420,
        rdzv_endpoint="0.0.0.0:12345",
    )

    # first train the model with p07
    p10_data_path = "/mnt/nvme1n1/datasets/data-to-export/BMO/bmo_skills.jsonl"
    # p07_ckpt_output_dir = "/mnt/7TB-a/models/granite-3.1-8b-v2-BMO-p07_muon-1e4"
    p10_ckpt_output_dir = "/mnt/7TB-a/models/granite-3.1-8b-v2-BMO-p10_muon-2e5"
    p10_input_model = (
        "/mnt/7TB-a/models/granite-3.1-8b-v2-BMO-p07_muon/hf_format/samples_64863"
    )
    max_batch_len = 45_000
    max_seq_len = 40_000

    # p10_data_path = "/mnt/7TB-a/datasets/ad-bmo-generated-04172025/skills_train_msgs_2025-04-17T19_54_20.jsonl"
    # p10_ckpt_output_dir = "/mnt/nvme1n1/models/granite-3.1-8b-v2-BMO-p10"
    # p10_input_model = None

    # set up the p07 training args
    p10_training_args = TrainingArgs(
        data_path=p10_data_path,
        model_path=p10_input_model,
        data_output_dir="/dev/shm",
        ckpt_output_dir=p10_ckpt_output_dir,
        max_batch_len=max_batch_len,
        max_seq_len=max_seq_len,
        num_epochs=7,
        effective_batch_size=3840,
        warmup_steps=0,
        save_samples=0,
        # learning_rate=1e-4,
        learning_rate=2e-5,
        checkpoint_at_epoch=True,
        accelerate_full_state_at_epoch=False,
        use_liger=False,
        distributed_backend="fsdp",
        fsdp_options=FSDPOptions(sharding_strategy=ShardingStrategies.HYBRID_SHARD),
        # Enable TensorBoard logging
        logger_type="tensorboard",  # or "async,tensorboard" for both
        run_name="granite-3.1-v2.4_muon_{time}",  # Will include timestamp
        optimizer="Muon",
    )
    run_training(torchrun_args, p10_training_args)

    # # set up the p10 training args
    # p10_input_model = None
    # p07_checkpoint_location = f"{p07_ckpt_output_dir}/hf_format"
    # p07_checkpoints = os.listdir(p07_checkpoint_location)
    # most_recent_checkpoint, most_recent_time = None, 0
    # for checkpoint in p07_checkpoints:
    #     full_ckpt_path = f"{p07_checkpoint_location}/{checkpoint}"
    #     if os.stat(full_ckpt_path).st_ctime > most_recent_time:
    #         most_recent_checkpoint = full_ckpt_path
    #         most_recent_time = os.stat(full_ckpt_path).st_ctime

    # # print out the most recent checkpoint
    # print(f"best checkpoint: {most_recent_checkpoint}")
    # p10_input_model = most_recent_checkpoint
    # p10_data_path = "/mnt/7TB-a/datasets/ad-bmo-generated-04172025/skills_train_msgs_lab_v2_bmo.jsonl"
    # p10_ckpt_output_dir = "/mnt/nvme1n1/models/granite-3.1-8b-v2-BMO-p10"
    # assert p10_input_model is not None

    # # set up the p10 training args
    # p10_training_args = TrainingArgs(
    #     data_path=p10_data_path,
    #     model_path=p10_input_model,
    #     data_output_dir="/dev/shm",
    #     ckpt_output_dir=p10_ckpt_output_dir,
    #     max_batch_len=max_batch_len,
    #     max_seq_len=max_seq_len,
    #     num_epochs=7,
    #     effective_batch_size=3840,
    #     warmup_steps=25,
    #     save_samples=0,
    #     learning_rate=2e-5,
    #     checkpoint_at_epoch=True,
    #     accelerate_full_state_at_epoch=True,
    #     use_liger=False,
    #     distributed_backend="fsdp",
    #     fsdp_options=FSDPOptions(sharding_strategy=ShardingStrategies.HYBRID_SHARD),
    # )
    # run_training(torchrun_args, p10_training_args)


if __name__ == "__main__":
    granite_31_8b_v2_bmo_p10()
