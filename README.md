# A Bitter Lesson for Data Filtering
Christopher Mohri, John Duchi, Tatsunori Hashimoto

The code for the paper is mostly contained in two external repositories:
* **Training:** [Meta Lingua](https://github.com/facebookresearch/lingua)
* **Filtering:** [DCLM](https://github.com/mlfoundations/dclm)

We mainly provide config files and instructions here.

---

## Replication Steps

To replicate the experiments at a given pool size:

### 1. Data and Filtering

*Note: You can skip the data sampling and filtering step entirely and just download the pre-processed files from [this Google Drive folder](https://drive.google.com/drive/folders/17QaXrff2dV-UOYoCmeHDAwwVsuLQvkDy).*

Take a random sample from [DCLM-Pool](https://data.commoncrawl.org/contrib/datacomp/index.html) of the desired pool size. Then, follow the [DCLM baselines instructions](https://github.com/mlfoundations/dclm/blob/main/baselines/README.md) to apply the filters. Use the corresponding configs provided below:

* **RefinedWeb:** `dclm_baseline_refinedweb.yaml` *(in DCLM repo)*
* **English filter only:** `bitter_lesson_code/dclm/baselines/baselines_configs/just_english_filter.yaml`
* **Repetition filter only:** `bitter_lesson_code/dclm/baselines/baselines_configs/just_repetition_filter.yaml`
* **Stop word filter only:** `bitter_lesson_code/dclm/baselines/baselines_configs/just_stop_word_filter.yaml`
* **DCLM-Baseline:** Create RefinedWeb, follow dedup instructions in the DCLM repo, and use `fasttext_filter.yaml`
* **+random:** Run `bitter_lesson_code/generate_random_memory_efficient.py` *(specify location of pool data and growth target)*
* **+shuffled:** Run `bitter_lesson_code/generate_shuffle_memory_efficient.py` *(specify location of pool data and growth target)*

### 2. Training
We use `bitter_lesson_code/lingua/apps/main/scripts/filtering_sweep_combined.py` to launch training sweeps. Fill in the constants at the top of the file (`PARTITION`, `BASE_DIR`, `CONDA_PATH`, `CONDA_ENV_PATH`, `EMAIL`) before running and edit `bitter_lesson_code/lingua/apps/main/configs/miso_8_combined.slurm` and `bitter_lesson_code/lingua/apps/main/scripts/utils.py` for your cluster/training setup. We provide our configs in the `bitter_lesson_code/lingua/apps/main/configs` folder; note that `bitter_lesson_code/lingua/apps/main/scripts/filtering_sweep_combined.py` overrides some of the values.

Each config file contains `!!!CHANGE_THIS!!!` placeholders that must be filled in: `data.root_dir` and `eval.validation.root_dir` (paths to your tokenized data), `data.tokenizer.path` (the tiktoken `r50k_base` tokenizer is included at `bitter_lesson_code/r50k_base_tokenizer/0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4`), and `slurm.account` / `slurm.partition`. 