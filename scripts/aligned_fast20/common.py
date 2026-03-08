from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACE_ROOT = REPO_ROOT / "ace"
AB_ROOT = REPO_ROOT / "AbstentionBench"

FAST_INDICES_PATH = AB_ROOT / "data" / "fast-subset-indices.json"
ALIGNED_DATA_DIR = ACE_ROOT / "eval" / "abstention" / "data" / "aligned_fast20"
MANIFEST_PATH = ALIGNED_DATA_DIR / "manifest.json"

AB_RESULTS_ROOT = AB_ROOT / "results" / "aligned_fast20"
ACE_QUICK_RESULTS_ROOT = ACE_ROOT / "results" / "aligned_fast20_quick"
ACE_FULL_RESULTS_ROOT = ACE_ROOT / "results" / "aligned_fast20_full"

REPORT_DIR = REPO_ROOT / "scripts" / "aligned_fast20" / "reports"

EXCLUDED_DATASETS = {"AveritecDataset"}

DATASET_TO_ACE_TASK = {
    "ALCUNADataset": "alcuna_fast_online",
    "BBQDataset": "bbq_fast_online",
    "BigBenchDisambiguateDataset": "big_bench_disambiguate_fast_online",
    "CoCoNotDataset": "coconot_fast_online",
    "FalseQADataset": "falseqa_fast_online",
    "FreshQADataset": "freshqa_fast_online",
    "GSM8K": "gsm8k_fast_online",
    "KUQDataset": "kuq_fast_online",
    "MediQDataset": "mediq_fast_online",
    "MMLUMath": "mmlu_math_fast_online",
    "MoralChoiceDataset": "moralchoice_fast_online",
    "MusiqueDataset": "musique_fast_online",
    "NQDataset": "nq_fast_online",
    "QAQADataset": "qaqa_fast_online",
    "QASPERDataset": "qasper_fast_online",
    "SelfAwareDataset": "selfaware_fast_online",
    "SituatedQAGeoDataset": "situatedqa_geo_fast_online",
    "Squad2Dataset": "squad2_fast_online",
    "UMWP": "umwp_fast_online",
    "WorldSenseDataset": "worldsense_fast_online",
}

