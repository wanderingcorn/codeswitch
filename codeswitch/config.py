"""Central configuration. Edit here; the notebook uses %autoreload so changes apply on re-run."""

IGNORE = -100          # label value ignored by the losses

# ---- Precision ----
# V100 has no bf16 / TF32: use fp16 autocast + GradScaler (fp32 master weights). Auto-disabled off CUDA.
USE_AMP = True
SEED = 42

# ---- Data ----
DATA_PATH = "SwitchLingua_text.csv"
BAD_PATH = "parse_failed_conversations.txt"

# (first_language, second_language) -> pair tag. Names must match the CSV values exactly.
ALL_PAIRS = {
    ("Hindi", "English"):      "hi-en",
    ("Cantonese", "English"):  "yue-en",
    ("Spanish", "English"):    "es-en",
    ("Arabic", "English"):     "ar-en",
    ("German", "English"):     "de-en",
    ("Vietnamese", "English"): "vi-en",
    ("Japanese", "English"):   "ja-en",
    ("Korean", "English"):     "ko-en",
    ("French", "English"):     "fr-en",
    ("Russian", "English"):    "ru-en",
    ("Italian", "English"):    "it-en",
    ("Chinese", "English"):    "zh-en",
}
PAIR_SEL = ["yue-en"]          # <- override this from the grid

def apply_pair_sel():
    global PAIRS, PAIR_TAGS, PAIR2ID
    sel = list(ALL_PAIRS.values()) if PAIR_SEL == "all" else PAIR_SEL
    PAIRS = {k: v for k, v in ALL_PAIRS.items() if v in sel}
    PAIR_TAGS = list(PAIRS.values())
    PAIR2ID = {t: i for i, t in enumerate(PAIR_TAGS)}

apply_pair_sel()
MAX_CONV_PER_PAIR = 10000   # max conversations per pair; None = use all
L1_MIN_SHARE = 0.1        # corpus_report warns when the L1 unit share falls below this
MIN_UNITS = 2             # a sentence needs at least this many units to have a prediction point
VAL_FRAC = 0.1            # fraction of conversations (per pair) held out for validation

# ---- Labels ----
N_DUR = 4                 # 0 = no switch; 1: 1-2 units; 2: 3-7; 3: >=8
def dur_class(run):
    """Duration class for a new segment of `run` units (1..3)."""
    return 1 + (run > 2) + (run > 7)

LAMBDA_DUR = 1          # weight of the duration loss

# ---- XLM-R ----
XLMR_NAME = "xlm-roberta-base"
MAX_LEN = 128             # safety truncation (left side is dropped)
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-5
HEAD_LR = 1e-4
KEEP_NEG = 1.0            # fraction of switch=0 training examples kept (switch=1 always kept); 1.0 = no downsampling (class weights handle the imbalance)
BUCKET_MULT = 50          # length bucketing: sort within chunks of BATCH_SIZE * BUCKET_MULT examples
VAL_BATCH_SIZE = 64

# ---- Qwen3 ----
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
QWEN_MAX_LEN = 256        # Devanagari takes many Qwen tokens, so this is looser than MAX_LEN
QWEN_BATCH_SIZE =  64      # On P100-12GB, use 8. On Macbook, use 16. On V100-32GB (fp16 AMP), 64 is safe; try 128.
QWEN_EPOCHS = 4
QWEN_LR = 5e-6
QWEN_HEAD_LR = 1e-4
DUR_SMOOTH = 0.1          # label smoothing for the duration loss
LAMBDA_ORD = 0.5          # ordinal L1 penalty on the expected duration class
