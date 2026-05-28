# RETROFIT

The repository contains a complete implementation of our proposed continual learning framework RETROFIT, along with all reproducible workflows necessary to adapt, merge, and evaluate models in the two binary security applications described in the paper, i.e., malware detection and binary summarization.  

For more details, please refer to our paper `RETROFIT: Continual Learning with Controlled Forgetting for Binary Security Detection and Analysis`

## MALWARE DETECTION

### Data

The raw APK samples used in our malware-detection experiments come from the Transcendent dataset and should be requested from the original authors via the project page: [link](https://s2lab.cs.ucl.ac.uk/projects/transcend/). To facilitate reproducibility and avoid redistributing raw APKs, we provide the extracted Drebin features used by our training and evaluation scripts under `Malware_Detection/data/transcendent/`, where you can find raw monthly feature files in:

```text
Malware_Detection/data/transcendent/YYYY-MM/features.pkl
```

Each `features.pkl` must contain:

- `json_features`
- `label`

### Install

```bash
pip install -r Malware_Detection/requirements.txt
```

### Run

Full pipeline:

```bash
python Malware_Detection/run_all.py all
```

Single stages:

```bash
python Malware_Detection/run_all.py prepare
python Malware_Detection/run_all.py train
python Malware_Detection/run_all.py eval
python Malware_Detection/run_all.py draw
```

## BINARY ANALYSIS

### Data and Model

For the binary-summarization experiments, we use the BinT5 dataset and base model released by the CAPYBARA project; the original [dataset](https://huggingface.co/datasets/AISE-TUDelft/Capybara) and [model](https://huggingface.co/collections/AISE-TUDelft/bint5) are publicly available on Hugging Face. We additionally release our fine-tuned checkpoints on Hugging Face: [link](https://huggingface.co/collections/SheHongyu/codet5-capybara-retrofit) 

### Setup
We build our replication package upon the [BinT5](https://github.com/AISE-TUDelft/Capybara-BinT5) environment. Follow the steps below inside your workspace to initialize the container and download the external artifacts:

1. **Docker Environment**: Pull the image and run the container:
```bash
docker pull aalkaswan/bint5
docker run -i -t --name retrofit_binary --gpus all -v $(pwd):/data aalkaswan/bint5 /bin/bash
cd /data/
```

You can re-enter the shell later using:

```bash
docker exec -it retrofit_binary /bin/bash
```

2. **Codebase Setup**: Clone the official CodeT5 repo into this directory:

```bash
git clone https://github.com/salesforce/CodeT5.git
```

3. **Data & Base Model Preparation**: Download the Capybara dataset and the reference `CodeT5-base` model:

```bash
wget https://zenodo.org/record/7229809/files/Capybara.zip
unzip Capybara.zip && rm Capybara.zip
mkdir -p CodeT5/CodeT5/data/summarize/{C,decomC,demiStripped,strippedDecomC}

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Salesforce/codet5-base
```

4. **Path Setup**:

- `CODET5_SH_DIR`: path to `CodeT5/CodeT5/sh`
- `--base_model_path`: local `codet5-base` or `CodeT5-C` directory
- `--adapter_paths`: adapter directories produced during training
- `--val_files`: validation jsonl files used during merge

### Train

Our method starts from the fully fine-tuned [CodeT5-C model](https://huggingface.co/SheHongyu/CodeT5-C).

#### Step 1: Data Placement

Prepare your data by moving it from the unpacked Capybara folder to the CodeT5 project directory:

```bash
cp -r ./Capybara/training_data/C/dedup/* ./CodeT5/CodeT5/data/summarize/C/
```

#### Step 2: Full Fine-Tuning to Obtain CodeT5-C

Before training our adapter, obtain the `CodeT5-C` initial model by fully fine-tuning `CodeT5-base` on the `C` dataset:

```bash
bash /data/job.sh
```

#### Step 3: LoRI Adapter Injection & Training

Replace the official CodeT5 files with our supplied low-rank implementations:

```bash
cp Binary_Analysis/Util/models_low_rank.py ./CodeT5/CodeT5/models.py
cp Binary_Analysis/Util/run_gen_low_rank.py ./CodeT5/CodeT5/run_gen.py
```

Then run:

```bash
CODET5_SH_DIR=/path/to/CodeT5/CodeT5/sh bash Binary_Analysis/CodeT5_train.sh
```

### Merge

After training the adapter for the new task, run:

```bash
bash Binary_Analysis/CodeT5_merge.sh \
  --base_model_path /path/to/base_model \
  --adapter_paths /path/to/adapter_task1 /path/to/adapter_task2 \
  --val_files /path/to/task1-valid.jsonl /path/to/task2-valid.jsonl \
  --out_adapter_dir /path/to/output_dir \
  --out_full_pt_path /path/to/output_dir/merged_state_dict.pt
```

If you want a standard merged HuggingFace checkpoint afterwards:

```bash
python Binary_Analysis/Util/merge_and_unload_adapter.py \
  --base_model_path /path/to/base_model \
  --adapter_path /path/to/adapter \
  --save_dir /path/to/save_dir
```

### Eval

Run the official test code of CodeT5:

```bash
CODET5_SH_DIR=/path/to/CodeT5/CodeT5/sh bash Binary_Analysis/CodeT5_test.sh
```

If you wish to compute additional metrics:

```bash
python Binary_Analysis/Eval_Metrics.py --base_dir Binary_Analysis/Output
```

Besides BLEU reported in the main experiments of our paper, we provide supplementary summarization benchmark results [here](https://github.com/XdlSecZJU/RETROFIT-CL/blob/main/Binary_Analysis/complementary_summarization_metric.pdf).

## License

This project is licensed under the Apache 2.0 License.
