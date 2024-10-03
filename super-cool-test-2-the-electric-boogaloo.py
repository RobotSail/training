# First Party
from instructlab.training import (
    DeepSpeedOptions,
    FSDPOptions,
    LoraOptions,
    ShardingStrategies,
    TorchrunArgs,
    TrainingArgs,
    DataProcessArgs,
    run_training,
)
from instructlab.training.utils import get_training_cmd
from argparse import ArgumentParser

# torchargs = TorchrunArgs(
#     nproc_per_node=2,
#     nnodes=1,
#     node_rank=0,
#     rdzv_endpoint="127.0.0.1:12345",
#     rdzv_id=12345,
# )

# trainargs = TrainingArgs(
#     chat_tmpl_path="/home/ec2-user/training/src/instructlab/training/chat_templates/ibm_generic_tmpl.py",
#     ckpt_output_dir="checkpoints",
#     data_output_dir="/dev/shm",
#     data_path="/home/ec2-user/train_msgs.jsonl",
#     disable_flash_attn=False,
#     effective_batch_size=128,
#     is_padding_free=False,
#     lora=LoraOptions(
#         alpha=32,
#         dropout=0.1,
#         quantize_data_type="nf4",
#         rank=2,
#         target_modules=["k_proj", "o_proj", "c_proj"],
#     ),
#     deepspeed_options=DeepSpeedOptions(
#         cpu_offload_optimizer=True,
#         cpu_offload_optimizer_pin_memory=True,
#         cpu_offload_optimizer_ratio=1,
#     ),
#     save_samples=500,
#     learning_rate=2e-5,
#     max_batch_len=20_000,
#     max_seq_len=1000,
#     mock_data=False,
#     mock_data_len=0,
#     model_path="/home/ec2-user/.cache/instructlab/models/instructlab/granite-7b-lab",
#     num_epochs=1,
#     random_seed=128,
#     warmup_steps=25,
# )


# torchargs = TorchrunArgs(
#     nproc_per_node=4,
#     nnodes=1,
#     node_rank=0,
#     rdzv_endpoint="127.0.0.1:12345",
#     rdzv_id=12345,
# )

# trainargs = TrainingArgs(
#     chat_tmpl_path="/home/ec2-user/training/src/instructlab/training/chat_templates/ibm_generic_tmpl.py",
#     ckpt_output_dir="checkpoints",
#     data_output_dir="/dev/shm",
#     data_path="/home/ec2-user/training/sample-data/train_all_pruned_SDG.jsonl",
#     disable_flash_attn=False,
#     effective_batch_size=3840,
#     is_padding_free=True,
#     deepspeed_options=DeepSpeedOptions(
#         cpu_offload_optimizer=True,
#         cpu_offload_optimizer_pin_memory=True,
#         cpu_offload_optimizer_ratio=1,
#     ),
#     fsdp_options=FSDPOptions(
#         offload_params=True, sharding_strategy=ShardingStrategies.FULL_SHARD
#     ),
#     lora=LoraOptions(
#         target_modules=["c_proj"],
#         rank=4,
#     ),
#     distributed_backend="fsdp",
#     save_samples=450,
#     learning_rate=1e-6,
#     # FSDP max
#     # max_batch_len=2722,
#     # max_seq_len=2722,
#     max_batch_len=4096,
#     max_seq_len=4096,
#     mock_data=False,
#     mock_data_len=0,
#     model_path="/home/ec2-user/.cache/instructlab/models/instructlab/granite-7b-lab",
#     # model_path="/home/ec2-user/training/checkpoints/hf_format/samples_384",
#     num_epochs=1,
#     random_seed=128,
#     warmup_steps=25,
# )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--padding_free", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        from instructlab.training.main_ds import get_script_name
        from instructlab.training.data_process import main as dp_main

        dp_main(
            DataProcessArgs(
                # XXX(osilkin): make a decision here, either:
                #   1. the CLI is fully responsible for managing where the data is written
                #   2. we never cache it and simply write it to a tmp file every time.
                #
                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
                # where the user has a defined place for new temporary data to be written.
                data_output_path=trainargs.data_output_dir,
                model_path=trainargs.model_path,
                data_path=trainargs.data_path,
                max_seq_len=trainargs.max_seq_len,
                chat_tmpl_path=trainargs.chat_tmpl_path,
            )
        )

        cmd = get_training_cmd(
            torch_args=torchargs, train_args=trainargs, script=get_script_name()
        )
        print(" ".join(cmd))
    else:
        run_training(torch_args=torchargs, train_args=trainargs)
