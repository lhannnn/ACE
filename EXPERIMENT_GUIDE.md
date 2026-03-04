# ACE vs AbstentionBench Experiment Guide

## Project Overview

This project compares two frameworks for LLM abstention (refusing to answer when appropriate):

- **AbstentionBench** (`AbstentionBench/`): A benchmark that evaluates whether LLMs know when to abstain from answering. It has 21 datasets across 6 abstention scenarios, each with 100-sample fast-subsets.
- **ACE** (`ace/`): A playbook-based learning framework (Generator → Reflector → Curator loop) that iteratively improves LLM performance via a "playbook" of strategies.

We have already integrated ACE with AbstentionBench's datasets and run initial experiments on UMWP (100 samples) using Together.ai API with `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`. The goal now is to **run ALL 21 datasets with a locally deployed model** instead of calling an API.

---

## Repository Structure

```
ACE/
├── ace/                              # ACE framework
│   ├── ace/                          # Core ACE code (Generator, Reflector, Curator)
│   │   ├── ace.py                    # Main orchestrator (run, _online_train_and_test, _offline_train)
│   │   ├── core/                     # Agent implementations
│   │   └── prompts/                  # Prompt templates (generator.py, reflector.py, curator.py)
│   ├── utils.py                      # initialize_clients(), evaluate_test_set(), extract_answer()
│   ├── llm.py                        # LLM call utilities
│   ├── .env                          # API keys (TOGETHER_API_KEY, etc.)
│   ├── eval/
│   │   ├── finance/                  # Reference task implementation
│   │   └── abstention/              # ★ Our abstention task integration
│   │       ├── data_processor.py     # DataProcessor with abstention-aware evaluation
│   │       ├── run.py                # Entry point for running ACE on abstention tasks
│   │       └── data/                 # Exported JSONL data + config
│   │           ├── abstention_config.json
│   │           ├── umwp_all.jsonl (100), umwp_train.jsonl (60), umwp_val.jsonl (20), umwp_test.jsonl (20)
│   │           ├── selfaware_*.jsonl
│   │           └── falseqa_*.jsonl
│   └── results/                      # ACE experiment results
│
├── AbstentionBench/                  # AbstentionBench framework
│   ├── main.py                       # Main entry (Hydra-based)
│   ├── recipe/
│   │   ├── models.py                 # Model classes (TogetherAIAPI, TogetherAIAPI_ACEFormat, etc.)
│   │   ├── evaluation.py             # LLM Judge abstention detection + evaluation
│   │   ├── evaluation_judge_prompts.py  # Judge prompt templates
│   │   └── abstention_datasets/      # Dataset loaders (one .py per dataset)
│   ├── configs/
│   │   ├── model/                    # Model configs (together_ai.yaml, together_ai_ace_format.yaml)
│   │   ├── dataset/                  # Dataset configs (umwp.yaml, self_aware.yaml, etc.)
│   │   └── abstention_detector/      # Judge configs
│   ├── data/
│   │   └── fast-subset-indices.json  # 100-sample subsets for each of 21 datasets
│   └── results/                      # AbstentionBench experiment results
│
└── EXPERIMENT_GUIDE.md               # ★ This file
```

---

## The 21 Datasets (6 Abstention Scenarios)

| Scenario | Datasets | Config name |
|---|---|---|
| **Underspecified Context** | UMWP, GSM8K, MMLUMath, GPQA, WorldSense, BBQ, BigBenchDisambiguate | umwp, gsm8k, mmlu_math, gpqa, worldsense, bbq, big_bench_disambiguate |
| **Answer Unknown** | SelfAware, ALCUNA, BigBenchKnownUnknowns, KUQ, NQ, Squad2, MediQ | self_aware, alcuna, big_bench_known_unknowns, kuq, (no nq config?), squad2, mediq |
| **False Premise** | FalseQA, QAQA, FreshQA | falseqa, qaqa, freshqa |
| **Underspecified Intent** | SituatedQA, Musique, QASPER | situated_qa, musique, qasper |
| **Subjective** | MoralChoice | moralchoice |
| **Unsupported/Stale** | CoCoNot | coconot |

Each dataset has a 100-sample fast-subset defined in `data/fast-subset-indices.json`.

---

## What Needs to Change for Local Model Deployment

Currently both frameworks use Together.ai API. To switch to a locally deployed model:

### For ACE (`ace/`):

1. **`ace/utils.py`** — `initialize_clients()` creates OpenAI-compatible clients. For a local model served via vLLM/Ollama/TGI, change `base_url` to point to `http://localhost:PORT/v1` and set any dummy `api_key`.

2. **`ace/.env`** — Update with local endpoint info.

3. **`ace/eval/abstention/run.py`** — The `--api_provider` and `--generator_model` / `--reflector_model` / `--curator_model` args control the model. If serving locally via an OpenAI-compatible server:
   - You could add a new api_provider like "local" in `utils.py`
   - Or just modify the "together" provider to point to localhost

### For AbstentionBench (`AbstentionBench/`):

1. **`AbstentionBench/recipe/models.py`** — Contains model classes. The existing `TogetherAIAPI` and `TogetherAIAPI_ACEFormat` classes use the OpenAI client. Create a similar class pointing to the local model server, or modify the existing ones to accept a configurable `base_url`.

2. **`AbstentionBench/configs/model/`** — Create new YAML configs for the local model.

3. **`AbstentionBench/main.py`** — `MODELS_WITHOUT_GPU` list needs the new class name added.

---

## Experiment Matrix

For each of the 21 datasets, run these 4 conditions:

| # | Method | Framework | Description |
|---|---|---|---|
| 1 | **AbstentionBench (plain prompt)** | AbstentionBench | Direct question → model → LLM Judge evaluates abstention |
| 2 | **AbstentionBench (ACE JSON prompt)** | AbstentionBench | Question wrapped in ACE's generator prompt format → model → LLM Judge evaluates |
| 3 | **ACE online (pre-train)** | ACE | Online mode: playbook accumulates across samples. Pre-train = answer BEFORE learning from each sample |
| 4 | **ACE online (post-train)** | ACE | Same run as #3. Post-train = answer AFTER playbook updated for each sample (has seen ground truth, so inflated) |

Optional additional condition:
| 5 | **ACE offline** | ACE | Train on 60 samples, validate on 20, test on held-out 20 |

### Evaluation: ALL conditions must use the same LLM Judge

The AbstentionBench LLM Judge prompt (from `recipe/evaluation_judge_prompts.py`) classifies whether a model response is an abstention. For fair comparison, ACE results must ALSO be evaluated using this same LLM Judge on the **full model response** (not just the extracted `final_answer`).

Key finding from our experiments: Using only the extracted `final_answer` for judging gives much lower scores than using the full response, because the reasoning/uncertainty expressions in the full response are lost.

**Strict judge mode (recommended):**
- In ACE, keep `--strict_judge` enabled (default) so Judge failures are treated as **indeterminate** (`is_abstention=None`) instead of keyword fallback.
- Report both:
  - `total`: all samples
  - `evaluated_total`: samples with determinate judge outputs
- This keeps metric semantics aligned with AbstentionBench's indeterminate filtering behavior.

---

## How to Run Each Condition

### Condition 1: AbstentionBench (plain prompt)

```bash
cd AbstentionBench

# Set environment variable for local model OR API
export TOGETHER_API_KEY=your_key  # or configure local endpoint

# Run for each dataset (replace DATASET with: umwp, self_aware, falseqa, etc.)
python main.py -m \
  mode=local dataset=DATASET model=together_ai \
  abstention_detector=llm_judge_togetherai \
  run_single_job_for_inference_and_judge=True \
  common_dir=$(pwd) \
  dataset_indices_path=data/fast-subset-indices.json
```

### Condition 2: AbstentionBench (ACE JSON prompt)

Same as above but `model=together_ai_ace_format`:

```bash
python main.py -m \
  mode=local dataset=DATASET model=together_ai_ace_format \
  abstention_detector=llm_judge_togetherai \
  run_single_job_for_inference_and_judge=True \
  common_dir=$(pwd) \
  dataset_indices_path=data/fast-subset-indices.json
```

**Important**: The `ace_context_key` in `configs/model/together_ai_ace_format.yaml` must match the dataset's scenario. You'll need to create per-scenario YAML configs or make it dynamic (see `ACE_CONTEXT_MAP` in `recipe/models.py`).

### Conditions 3 & 4: ACE online (pre-train + post-train)

```bash
cd ace

# First: export data from AbstentionBench for each dataset (see Data Export section below)

python -m eval.abstention.run \
    --task_name TASK_NAME_online \
    --mode online \
    --api_provider together \
    --judge_api_provider together \
    --judge_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --strict_judge \
    --save_path results
```

Results are in `results/ace_run_TIMESTAMP/pre_train_post_train_results.json` (per-sample pre/post answers) and `detailed_llm_logs/` (full responses).

### Condition 5 (optional): ACE offline

```bash
python -m eval.abstention.run \
    --task_name TASK_NAME \
    --mode offline \
    --api_provider together \
    --save_path results
```

---

## Data Export: Adding New Datasets to ACE

Currently only 3 datasets (UMWP, SelfAware, FalseQA) are exported to ACE JSONL format. To add all 21 datasets:

1. First run AbstentionBench on the dataset (Condition 1) to generate the `GroundTruthAbstentionEvaluator.json` results file.

2. Export from results to JSONL:

```python
import json, random
random.seed(42)

# Read AbstentionBench results
with open('AbstentionBench/results/DATASET_MODEL/TIMESTAMP/GroundTruthAbstentionEvaluator.json') as f:
    data = json.load(f)

samples = []
for r in data['responses']:
    prompt = r['prompt']
    samples.append({
        "question": prompt["question"],
        "should_abstain": prompt["should_abstain"],
        "reference_answers": prompt.get("reference_answers"),
        "scenario": "SCENARIO_NAME",  # one of: underspecified_context, answer_unknown, false_premise, underspecified_intent, subjective, unsupported_stale
        "metadata": prompt.get("metadata", {})
    })

random.shuffle(samples)
# Write to ace/eval/abstention/data/DATASET_all.jsonl
# Also split into train/val/test (60/20/20)
```

3. Update `ace/eval/abstention/data/abstention_config.json` with the new dataset paths.

4. Update `ace/eval/abstention/run.py` to accept the new task_name in the `choices` list.

5. Update `ace/eval/abstention/data_processor.py` `_build_context()` to handle new scenarios (underspecified_intent, subjective, unsupported_stale).

---

## Post-Processing: Computing AbstentionBench Metrics from ACE Results

After running ACE, compute the same metrics using the LLM Judge on ACE's **full responses**:

```python
import json, os, glob, openai

# 1. Load original data (for question, reference_answers, should_abstain)
with open('ace/eval/abstention/data/DATASET_all.jsonl') as f:
    orig_data = [json.loads(line) for line in f if line.strip()]

# 2. Load full responses from ACE detailed logs
log_dir = 'ace/results/ace_run_TIMESTAMP/detailed_llm_logs'
pre_train_responses = {}
for f in glob.glob(f'{log_dir}/generator_online_train_s_*_gen_initial_*.json'):
    step = int(os.path.basename(f).split('_gen_initial_')[0].split('_s_')[1])
    with open(f) as fh:
        pre_train_responses[step] = json.load(fh)['response']

# 3. For each sample, call LLM Judge with the FULL response
# Use the same judge prompt from AbstentionBench/recipe/evaluation_judge_prompts.py
# Judge returns "Yes" (abstention) or "No" (not abstention)

# 4. Compute TP, FP, FN, TN → Precision, Recall, F1, Accuracy
```

---

## Key Metrics

For each condition on each dataset, report:
- **TP**: should_abstain=True AND model abstained (correct)
- **FP**: should_abstain=False AND model abstained (incorrect)
- **FN**: should_abstain=True AND model did NOT abstain (incorrect)
- **TN**: should_abstain=False AND model did NOT abstain (correct)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 * Precision * Recall / (Precision + Recall)
- **Accuracy** = (TP + TN) / Total

---

## Previous Results (UMWP, 100 samples, Llama 3.1 8B via Together.ai API)

All evaluated with the same LLM Judge on full model responses:

| Method | Prec | Recall | F1 | Acc |
|---|---|---|---|---|
| AbstentionBench (plain prompt) | 1.000 | 0.585 | 0.738 | 0.780 |
| AbstentionBench (ACE JSON prompt) | 0.967 | 0.547 | 0.699 | 0.750 |
| ACE pre-train (full response + LLM Judge) | 0.946 | 0.660 | 0.778 | 0.800 |
| ACE post-train (full response + LLM Judge) | 0.976 | 0.774 | 0.863 | 0.870 |

Note: ACE post-train is inflated because the model has seen ground truth for each sample during training. ACE pre-train is the fairer comparison with AbstentionBench.

---

## Important Notes

1. **ACE pre-train vs post-train**: In online mode, for each sample N, pre-train uses the playbook from samples 1..N-1 (never seen sample N's answer). Post-train uses the playbook updated from sample N (has seen the answer via Reflector). Pre-train is the fair metric.

2. **LLM Judge must use full response**: ACE's `extract_answer()` pulls a short `final_answer` from the JSON response. The LLM Judge MUST see the full response text to detect abstention signals in the reasoning.

3. **Curator JSON failures**: With 8B models, the Curator sometimes fails to produce valid JSON. ACE handles this gracefully by skipping the update. This is expected behavior, not a bug. Larger models (70B+) will have fewer failures.

4. **AbstentionBench has no train/test split**: All data is evaluation data. The train/val/test splits for ACE offline mode were created artificially (60/20/20 from the 100-sample subset).

5. **The `ace_context_key` in AbstentionBench ACE format**: Must match the scenario. Current `ACE_CONTEXT_MAP` in `models.py` has entries for: `umwp`, `selfaware`, `falseqa`. Need to add: `underspecified_intent`, `subjective`, `unsupported_stale`, etc.
