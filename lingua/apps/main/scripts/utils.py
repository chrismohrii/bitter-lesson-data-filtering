import os
import subprocess
import time
from dataclasses import dataclass
import shutil


@dataclass
class JobConfig:
    base_dir: str
    conda_path: str
    conda_env_path: str
    config_file: str
    email: str


def launch_combined_job(job_config_list: list, extra_args_list: list,  partition: str = 'miso'):
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env.update({
        "BASE_DIR": job_config_list[0].base_dir,
        "CONDA_PATH": job_config_list[0].conda_path,
        "CONDA_ENV_PATH": job_config_list[0].conda_env_path,
    })
    
    if partition == 'miso' or partition == 'miso-8':
        print('Using miso config')
        slurm_config = "apps/main/configs/miso_8_combined.slurm"
        nproc_per_node = 8
    else:
        raise ValueError(f"Unsupported partition for combined launch: {partition}")


    # Step 1: Backup original run_many.sh
    sh_path = slurm_config
    backup_path = f"{slurm_config}.bak"
    shutil.copyfile(sh_path, backup_path)

    # Step 2: Append new srun lines
    torchrun_cmds = [
        f"torchrun --nproc_per_node={nproc_per_node} -m apps.main.train config={job_config_list[i].config_file} {extra_args_list[i]}"
            for i in range(len(job_config_list))
    ]

    with open(sh_path, "a") as f:
        for cmd in torchrun_cmds:
            f.write(f"\nsrun -n 1 -N 1 -o out/logs/log_${{SLURM_JOB_ID}}.out -e out/logs/log_${{SLURM_JOB_ID}}.err {cmd}\n")


    # Launch sbatch command
    cmd = [
        "sbatch",
        f"--mail-user={job_config_list[0].email}",
        "--export=ALL,BASE_DIR,CONDA_PATH,CONDA_ENV_PATH",
        slurm_config
    ]

    print(cmd)

    subprocess.run(cmd, env=env)
    time.sleep(0.1)

    # Step 4: Restore original script
    shutil.move(backup_path, sh_path)

