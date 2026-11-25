"""Unit segmentation and language identification (LID).

Segmentation is regex-based. Per-unit labels: 0 = L1 (non-English), 1 = English.
LID runs ONCE per sentence (full-sentence context) and the result is cached in C.ANN_DIR/<pair>.jsonl:
  - hi-en                 : bhasha-lid-indic-v2 (word level; native + romanized)
  - zh/yue/ja/ko/ar/ru-en : script rule only (non-Latin unit = L1, Latin unit = English)
  - es/fr/it/de/vi-en     : MixLID
Training, evaluation and the demo only READ the cache (see data.load_records / data.load_convs).
"""
import json
import os
import re
import sys
import time
from typing import NamedTuple

from . import config as C

# ---------- Unit segmentation ----------
HAN    = r"[\u4e00-\u9fff\u3400-\u4dbf]"                      # Han: one character per unit (zh / yue / ja kanji)
KANA   = r"[\u3040-\u309f\u30a0-\u30ff]"                      # Japanese kana: one character per unit
DEV    = r"[\u0900-\u0963\u0971-\u097f][\u0900-\u0963\u0971-\u097f\u200c\u200d]*"   # Devanagari: one word
HANGUL = r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+"        # Korean: a run of Hangul is one word
CYR    = r"[\u0400-\u04ff]+(?:-[\u0400-\u04ff]+)*"            # Cyrillic: one word
ARA    = r"[\u0621-\u064a\u066e-\u06d3\u06fa-\u06ff][\u0621-\u064a\u0640\u064b-\u065f\u0670\u066e-\u06d3\u06fa-\u06ff]*"  # Arabic: one word
LATC   = "A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\u1e00-\u1eff"          # Latin letters incl. Vietnamese diacritics
LAT    = rf"[{LATC}]+(?:['’][{LATC}]+)*"

UNIT_RE = re.compile(f"{HAN}|{KANA}|{DEV}|{HANGUL}|{CYR}|{ARA}|{LAT}")
PIECE_RE = re.compile(r"\d+|[^\s\d]")      # a number or a single punctuation mark
LANG_NAME = {0: "L1", 1: "en"}

class Stop(NamedTuple):
    end: int        # 字符结束位置: text[:end] 即该 stop 的前缀
    is_unit: bool
    ref: int        # 标签取自的 unit 下标: unit 自身 / 前一个最近的 unit / 句首则取后一个 unit
    piece: str      # stop 的文本 (unit 词或标点/数字)

def find_stops(text):
    units = find_units(text)
    if not units:
        return []
    stops, pos = [], 0
    for i, u in enumerate(units):
        r = max(i - 1, 0)                     # unit i 之前的间隙: 句首(i=0)取 unit 0, 否则取前一个 unit
        stops += [Stop(m.end(), False, r, m.group()) for m in PIECE_RE.finditer(text, pos, u.start())]
        stops.append(Stop(u.end(), True, i, u.group()))
        pos = u.end()
    stops += [Stop(m.end(), False, len(units) - 1, m.group()) for m in PIECE_RE.finditer(text, pos)]   # 句尾
    return stops

def find_units(text):
    return list(UNIT_RE.finditer(text))


# ---------- Label maps ----------
HI_PAIR = "hi-en"
MIXLID_EN = "eng_Latn"
MIXLID_AUX = {"NUM", "PUNCT", "SYM", "und"}
_ZH = {"cmn_Hans", "cmn_Hant"}                                # MixLID has no Cantonese label: both Chinese labels = L1
MIXLID_L1 = {
    "zh-en": _ZH, "yue-en": _ZH, "ja-en": {"jpn_Jpan"} | _ZH, "ko-en": {"kor_Hang"},
    "ar-en": {"ara_Arab"}, "ru-en": {"rus_Cyrl"}, "es-en": {"spa_Latn"}, "fr-en": {"fra_Latn"},
    "it-en": {"ita_Latn"}, "de-en": {"deu_Latn"}, "vi-en": {"vie_Latn"},
}
BHASHA_NONLANG = {"name", "other"}                            # -> unknown (filled from the neighbour); any Indic label = L1
NON_LATIN_RE = re.compile(f"^(?:{HAN}|{KANA}|{DEV}|{HANGUL}|{CYR}|{ARA})")
L1_SCRIPT_PAIRS = {"zh-en", "yue-en", "ja-en", "ko-en", "ar-en", "ru-en"}   # 母语不是 Latin 书写的 pair


def fill_unknown(raw):
    """raw: list of 0 / 1 / None. None copies the previous unit's label (the first known one at the start; 1 if none).
    Returns (labels, number of unknown units)."""
    first = next((x for x in raw if x is not None), 1)
    out, prev = [], first
    for x in raw:
        prev = prev if x is None else x
        out.append(prev)
    return out, sum(x is None for x in raw)


# ---------- Taggers ----------
def _align(spans, toks):
    """spans: [(start, end)] 每个 unit stop 的字符范围。对重叠的子词累加 L1 / 英语概率, 取较大者。"""
    raw, j = [], 0
    for s, e in spans:
        while j < len(toks) and toks[j]["end"] <= s:
            j += 1
        l1_score, en_score, k = 0.0, 0.0, j
        while k < len(toks) and toks[k]["start"] < e:
            l1_score += toks[k]["l1"]; en_score += toks[k]["en"]; k += 1
        raw.append(None if l1_score == 0.0 and en_score == 0.0 else (0 if l1_score >= en_score else 1))
    return fill_unknown(raw)


class MixLIDTagger:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(C.MIXLID_NAME)
        self.model = AutoModelForTokenClassification.from_pretrained(C.MIXLID_NAME).to(device).eval()
        self.label2id = self.model.config.label2id
        self.en_id = self.label2id[MIXLID_EN]
        self.aux_ids = {self.label2id[k] for k in MIXLID_AUX if k in self.label2id}

    # MixLIDTagger.label: 原句输入，按 unit 取回，再用脚本覆盖
    def label(self, texts, pair):
        l1_ids = [self.label2id[k] for k in MIXLID_L1[pair] if k in self.label2id]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=C.MIXLID_MAX_LEN, return_offsets_mapping=True)
        offsets_batch = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu()
        out = []
        for t, offsets, tok_probs in zip(texts, offsets_batch, probs):
            units = find_units(t)
            spans = [(u.start(), u.end()) for u in units]
            toks = [{"start": s, "end": e, "l1": float(p[l1_ids].sum()), "en": float(p[self.en_id])}
                    for (s, e), p in zip(offsets, tok_probs) if s != e and int(p.argmax()) not in self.aux_ids]
            labs, unk = _align(spans, toks)
            out.append((labs, unk))
        return out


class BhashaTagger:
    def __init__(self, device):
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, os.path.dirname(hf_hub_download(C.BHASHA_NAME, "lid_wrapper.py")))
        from lid_wrapper import LID
        try:
            self.lid = LID(C.BHASHA_NAME, device=str(device))
        except TypeError:                                     # wrapper without a device argument
            self.lid = LID(C.BHASHA_NAME)

    def label(self, texts, pair):
        out = []
        for t in texts:
            stops = find_stops(t)
            labs = self.lid.tag([st.piece for st in stops]) if stops else []
            labs = [l[-1] if isinstance(l, (tuple, list)) else l for l in labs]
            raw = [1 if l == "en" else None if l in BHASHA_NONLANG else 0
                for l, st in zip(labs, stops) if st.is_unit]
            out.append(fill_unknown(raw))
        return out

class ScriptTagger:
    """Non-Latin L1 pairs: no model. Non-Latin script unit -> L1 (0); Latin unit -> English (1)."""
    def __init__(self, device=None):
        pass

    def label(self, texts, pair):
        return [([0 if NON_LATIN_RE.match(u.group()) else 1 for u in find_units(t)], 0) for t in texts]

_TAGGERS = {}


def get_tagger(pair):
    kind = "bhasha" if pair == HI_PAIR else "script" if pair in L1_SCRIPT_PAIRS else "mixlid"
    if kind not in _TAGGERS:
        from .utils import DEVICE
        _TAGGERS[kind] = {"bhasha": BhashaTagger, "script": ScriptTagger, "mixlid": MixLIDTagger}[kind](DEVICE)
    return _TAGGERS[kind]


# ---------- Disk cache ----------
def ann_path(pair):
    return os.path.join(C.ANN_DIR, f"{pair}.jsonl")


def load_ann(pair):
    """{"<row_id>:<sentence idx>": (n_units, "0110...")}"""
    out = {}
    if os.path.exists(ann_path(pair)):
        with open(ann_path(pair), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[d["k"]] = (d["n"], d["l"])
    return out


def lookup(ann, row_id, idx, n_units):
    """Cached unit labels (list of 0/1) or None if missing / the unit count no longer matches the segmentation."""
    r = ann.get(f"{row_id}:{idx}")
    if r is None or r[0] != n_units:
        return None
    return [int(c) for c in r[1]]


def annotate_items(pair, items, log=print):
    """items: [(key, cleaned text)]. Annotates what is not cached yet and appends to ANN_DIR/<pair>.jsonl (resumable)."""
    done = load_ann(pair)
    todo = [(k, t) for k, t in items if k not in done]
    log(f"[LID {pair}] {len(items)} sentences, {len(items) - len(todo)} cached, {len(todo)} to annotate")
    if not todo:
        return
    os.makedirs(C.ANN_DIR, exist_ok=True)
    tagger, t0, n_unk, n_units = get_tagger(pair), time.time(), 0, 0
    with open(ann_path(pair), "a", encoding="utf-8") as f:
        for i in range(0, len(todo), C.ANN_BATCH):
            batch = todo[i:i + C.ANN_BATCH]
            for (k, t), (langs, unk) in zip(batch, tagger.label([t for _, t in batch], pair)):
                f.write(json.dumps({"k": k, "n": len(langs), "l": "".join(map(str, langs)), "u": unk}) + "\n")
                n_unk += unk; n_units += len(langs)
            if (i // C.ANN_BATCH) % 50 == 0:
                f.flush()
                log(f"  {i + len(batch)}/{len(todo)} | {time.time() - t0:.0f}s")
    log(f"[LID {pair}] done; units with no usable label (filled from neighbour): {n_unk / max(1, n_units):.2%}")


# ---------- Self-test ----------
SELFTEST = [
    ("zh-en", "我喜欢 this movie，因为 it's 很好看。"),
    ("yue-en", "佢今日 meeting 完咗先返屋企。"),
    ("hi-en", "मुझे यह movie बहुत पसंद आई, because it was amazing!"),
    ("hi-en", "mujhe yeh movie bahut pasand aayi because it was amazing!"),   # romanized
    ("es-en", "Ayer fui al cine con mi amigo, and we watched a really good movie, pero no me gustó el final."),
    ("ar-en", "أنا أحب هذا movie كثيراً because it's great"),
    ("ja-en", "今日はmeetingがあるのでI'll be late"),
    ("ko-en", "나는 이 movie가 정말 좋아 because it was fun"),
    ("ru-en", "Я вчера watched a movie и мне понравилось"),
    ("vi-en", "Hôm nay tôi đi to the office rồi về nhà"),
    ("fr-en", "Je suis allé au cinema with my friend mais c'était nul"),
]


def selftest(pairs=None):
    """Print the (unit, label) sequence for one example sentence per pair (loads the LID models, no cache)."""
    for tag, t in SELFTEST:
        if pairs and tag not in pairs:
            continue
        langs, unk = get_tagger(tag).label([t], tag)[0]
        print(tag, [(u.group(), l) for u, l in zip(find_units(t), langs)], f"| unknown filled: {unk}")
