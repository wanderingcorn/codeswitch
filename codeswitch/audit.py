"""逐 stop 核查: python -m codeswitch.audit yue-en#65198 [--turn 3]
每个 stop 打印: piece | is_unit | ref | lang | switch | dur   (标签规则与 data.make_examples 完全一致)"""
import argparse

from . import config as C
from .data import load_convs, make_examples
from .lid import find_stops, find_units, LANG_NAME


def audit_turn(text, langs):
    units, stops = find_units(text), find_stops(text)
    n = len(units)
    ex = [e for e in make_examples(text, langs)]            # 与训练一致; 最后一个 unit 及其后的标点没有样本
    print(f"{'#':>3} {'piece':<14} {'unit':<5} {'ref':>3} {'lang':<5} {'sw':>2} {'dur':>3}  prefix 末尾")
    k = 0
    for j, s in enumerate(stops):
        lang = LANG_NAME[langs[s.ref]] if s.ref >= 0 else "-"
        if s.ref + 1 < n:                                    # 有标签的 stop
            e = ex[k]; k += 1
            assert e["text"] == text[:s.end]                 # 与 make_examples 对齐
            sw, du = str(e["switch"]), str(e["duration"])
        else:
            sw = du = "-"
        print(f"{j:>3} {s.piece:<14} {str(s.is_unit):<5} {s.ref:>3} {lang:<5} {sw:>2} {du:>3}  ...{text[max(0, s.end - 8):s.end]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("conv", help="例如 yue-en#65198")
    ap.add_argument("--turn", type=int, default=None, help="只看第几句 (默认全部)")
    a = ap.parse_args(argv)
    pair = a.conv.split("#")[0]
    C.PAIR_SEL = [pair]; C.apply_pair_sel()
    conv = load_convs([a.conv])[0]
    for t, turn in enumerate(conv["turns"]):
        if a.turn is not None and t != a.turn:
            continue
        print(f"\n=== {a.conv} turn {t}: {turn['text']}")
        if turn["langs"] is None:
            print("  (不足 MIN_UNITS 或没有缓存标注)")
            continue
        audit_turn(turn["text"], turn["langs"])


if __name__ == "__main__":
    main()