#!/usr/bin/env python3
from bitter_lesson_code.lingua.apps.main.scripts.utils import launch_combined_job, JobConfig

# --- Experiment Constants ---
# Fill these in for your cluster/training setup.
PROJECT_NAME = "bitter_lesson"
PARTITION = ""           # e.g. "miso-8"
BASE_DIR = ""            # repo base directory on the cluster
CONDA_PATH = ""          # path to conda binary
CONDA_ENV_PATH = ""      # path to the conda env to activate
EMAIL = ""               # slurm notification email
MISO_8_MULTIPLER = 32

# --- Sweep Definitions ---
# Each entry in data_list refers to a data source registered under
# data.root_dir (see configs/llama_*_dataless.yaml).
data_list = [
    "dclm_pool_1b_1x_1B_sample_shuffled",
    "refinedweb_1b_1x_1B_sample_shuffled",
]
# Pool-size suffix used in run names (e.g. "1B", "10B", "30pct", ...).
suffix = "1B"

model_configs = {
    "15M":  "apps/main/configs/llama_15M_8H200_dataless.yaml",
    "80M":  "apps/main/configs/llama_80M_8H200_dataless.yaml",
    "330M": "apps/main/configs/llama_330M_8H200_dataless.yaml",
    "1B":   "apps/main/configs/llama_1B_8H200_dataless.yaml",
    "7B":   "apps/main/configs/llama_7B_8H200_dataless.yaml",
}

step_counts   = [20_000]   # varied from 20k up to 6.4M in the paper
weight_decays = [0.1, 0.5]
learning_rates = [5e-3]    # 1e-2 for 15M, 5e-3 for 80M/330M/1B, 1e-3 for 7B
num_evals = 5

if PARTITION == "miso-8":
    step_counts = [int(step // MISO_8_MULTIPLER) for step in step_counts]


# --- Partition Options ---
partition_map = {
    'miso-8': {
        'data.batch_size': 64,
        'distributed.dp_replicate': 8,
    },
}

# --- Launch Logic ---
for dataset in data_list:
    if dataset.startswith("just_"):
        short_name = dataset.replace("just_", "").replace(f"_{suffix}", "")
        short_data = f"{short_name}_{suffix}"
    elif "pool" in dataset:
        short_data = f"pool_{suffix}"
    elif "refined" in dataset:
        short_data = f"refined_{suffix}"
    else:
        short_data = f"{dataset}_{suffix}"

    job_config_list = []
    extra_args_list = []

    for model_name, config_path in model_configs.items():
        for steps in step_counts:
            for wd in weight_decays:
                for lr in learning_rates:

                    job_config = JobConfig(BASE_DIR, CONDA_PATH, CONDA_ENV_PATH, config_path, EMAIL)

                    multiplier = MISO_8_MULTIPLER if PARTITION == 'miso-8' else 1
                    run_name = f"{short_data}_{model_name}_s{(steps * multiplier)//1000}k_wd{wd}_lr{lr}"

                    if model_name == "7B":
                        grad_acc_steps = 4
                    else:
                        grad_acc_steps = 1

                    overrides = [
                        f"data.sources.{dataset}=100.0",
                        f"optim.lr={lr}",
                        f"optim.weight_decay={wd}",
                        f"steps={steps}",
                        f"grad_acc_steps={grad_acc_steps}",
                        f"data.batch_size={partition_map[PARTITION]['data.batch_size'] // grad_acc_steps}",
                        f"distributed.dp_replicate={partition_map[PARTITION]['distributed.dp_replicate']}",
                        f"checkpoint.dump.every={max(1, int(steps//num_evals))}",
                        f"checkpoint.eval.every={max(1, int(steps//num_evals))}",
                        "eval.wipe_ckpt=false",
                        "distributed.spawn_method=spawn",
                        "slurm.dirs_exists_ok=true",
                    ]

                    extra_args = (
                        f"dump_base=out/{PROJECT_NAME} "
                        f"name={run_name} "
                        f"logging.wandb.project={PROJECT_NAME} "
                        f"logging.wandb.group={short_data}_{model_name} "
                        f"logging.wandb.tags=[{short_data},{model_name}] "
                        f"{' '.join(overrides)} "
                        f"seed=1 model.seed=1 data.seed=1"
                    )

                    job_config_list.append(job_config)
                    extra_args_list.append(extra_args)

    if job_config_list:
        print(f"Packaging {len(job_config_list)} runs for {short_data} into one job...")
        launch_combined_job(job_config_list, extra_args_list, partition=PARTITION)

print(f"Successfully submitted variants on {PARTITION}.")
