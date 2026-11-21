"""Interactive demo (MacBook-friendly): python -m codeswitch.demo --run out/NAME [--first-language Spanish] [--cs-type X]

1. Pick 10 complete TEST conversations, filtered by the CSV columns first_language / cs_type.
2. Type any prefix of any turn shown above -> P(switch) + hard decision, P(duration) + hard class.
3. The whole turn is then shown with its (cached) LID segmentation and the gold labels at that prefix.
Commands: :new = 10 new conversations | :filter = change the filter | :q = quit. Repeat as often as you like."""
import argparse
import json
import os
import random

from . import config as C
from .data import clean, load_convs, make_examples
from .lid import find_units
from .store import load_run


def _ask(msg, options=None):
    if options:
        print(f"  options: {sorted(options)}")
    try:
        return input(msg).strip()
    except EOFError:
        return ":q"


def pick(convs, first_language, cs_type, n, rng):
    pool = [c for c in convs if (not first_language or c["first_language"] == first_language)
            and (not cs_type or str(c["cs_type"]) == cs_type)]
    return rng.sample(pool, min(n, len(pool)))


def show_dialogues(shown):
    for i, c in enumerate(shown):
        print(f"\n=== Conversation {i} | {c['first_language']} | cs_type={c['cs_type']} | {c['conv']} ===")
        for t, turn in enumerate(c["turns"]):
            print(f"  [{i}.{t}] {turn['text']}")


def find_turns(shown, prefix):
    return [(i, t) for i, c in enumerate(shown) for t, turn in enumerate(c["turns"])
            if turn["langs"] is not None and turn["text"].startswith(prefix)]


def segments(text, langs):
    """Maximal same-language runs: (label, first unit, last unit, char start, char end)."""
    units, segs, i = find_units(text), [], 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and langs[j + 1] == langs[i]:
            j += 1
        segs.append((langs[i], i, j, units[i].start(), units[j].end()))
        i = j + 1
    return segs


def show_turn(conv, turn, prefix):
    names = {0: conv["first_language"], 1: "English"}
    text, segs = turn["text"], segments(turn["text"], turn["langs"])
    last = max(k for k, s in enumerate(segs) if s[3] < len(prefix))      # segment holding the end of the prefix
    print(f"\nWhole turn ({conv['conv']}), LID segmentation:")
    print(f"  {'#':>2} {'lang':<10} {'units':>5} {'dur':>4}  text")
    for k, (lg, a, b, s, e) in enumerate(segs):
        dur = "-" if k == 0 else str(C.dur_class(b - a + 1))             # class of the segment a switch leads into
        mark = "   <-- prefix ends here" if k == last else ""
        print(f"  {k:>2} {names[lg]:<10} {b - a + 1:>5} {dur:>4}  {text[s:e]}{mark}")


def show_prediction(p, gold):
    d = C.DUR_DESC
    print(f"\nPrediction (thr={p['thr']:.3f}):")
    print(f"  switch   P(switch)={p['p_switch']:.3f} -> {'SWITCH' if p['switch'] else 'no switch'}"
          f"   (argmax@0.5: {'SWITCH' if p['p_switch'] >= 0.5 else 'no switch'})")
    probs = "  ".join(f"P({d[k]})={v:.3f}" for k, v in zip((1, 2, 3), p["p_duration"]))
    print(f"  duration {probs} -> class {p['duration_if_switch']} ({d[p['duration_if_switch']]})"
          f"{'' if p['switch'] else '   [only meaningful if a switch happens]'}")
    if gold is None:
        print("  gold     n/a (the prefix does not end at a stopping point, or it is the last unit)")
        return
    gd = f", duration={gold['duration']} ({d[gold['duration']]})" if gold["switch"] else ""
    ok = "OK" if p["switch"] == gold["switch"] else "wrong"
    ok_d = "" if not gold["switch"] else (" duration OK" if p["duration_if_switch"] == gold["duration"] else " duration wrong")
    print(f"  gold     switch={gold['switch']}{gd}   -> switch {ok}{ok_d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="out/NAME of a finished xlmr / qwen run")
    ap.add_argument("--first-language"); ap.add_argument("--cs-type")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    kind, model, tok, thr, meta = load_run(a.run)
    if kind == "xlmr":
        from .xlmr import predict
    else:
        from .qwen import predict
    print(f"loaded {kind} model (pairs {meta['pairs']}, threshold {thr:.4f})")

    with open(os.path.join(a.run, "test_convs.json")) as f:
        convs = load_convs(json.load(f))
    convs = [c for c in convs if all(t["langs"] is not None or len(find_units(t["text"])) < C.MIN_UNITS for t in c["turns"])]
    rng = random.Random(a.seed)
    fl, ct = a.first_language, a.cs_type

    while True:
        shown = pick(convs, fl, ct, a.n, rng)
        if not shown:
            print(f"no test conversation for first_language={fl!r}, cs_type={ct!r}")
            fl = _ask("first_language: ", {c["first_language"] for c in convs}) or None
            ct = _ask("cs_type: ", {str(c["cs_type"]) for c in convs if not fl or c["first_language"] == fl}) or None
            continue
        show_dialogues(shown)
        while True:
            s = _ask("\nprefix of any turn above (:new / :filter / :q) > ")
            if s == ":q":
                return
            if s == ":new":
                break
            if s == ":filter":
                fl = _ask("first_language (Enter = any): ", {c["first_language"] for c in convs}) or None
                ct = _ask("cs_type (Enter = any): ", {str(c["cs_type"]) for c in convs if not fl or c["first_language"] == fl}) or None
                break
            p = clean(s)
            if not find_units(p):
                print("  type at least one word")
                continue
            hits = find_turns(shown, p)
            if len(hits) > 1:
                print("  matches:", ", ".join(f"[{i}.{t}]" for i, t in hits[:8]))
                pick_ = _ask("  which one (e.g. 2.3, Enter = first)? ")
                hits = [h for h in hits if f"{h[0]}.{h[1]}" == pick_] or hits[:1]
            pred = predict(model, tok, p, thr=thr)
            gold = None
            if hits:
                ci, ti = hits[0]
                turn = shown[ci]["turns"][ti]
                gold = {e["text"]: e for e in make_examples(turn["text"], turn["langs"])}.get(p)
            show_prediction(pred, gold)
            if hits:
                show_turn(shown[ci], turn, p)
            else:
                print("  (prefix not found in the turns above: no whole-turn annotation to show)")


if __name__ == "__main__":
    main()
