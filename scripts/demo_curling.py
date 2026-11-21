"""命令行演示:对冰壶比赛记录文法跑三种约束解码并生成 HTML 报告。

用法::

    python -m scripts.demo_curling            # 生成报告到 reports/curling.html
    python -m scripts.demo_curling --open     # 生成后尝试打开浏览器
"""

from __future__ import annotations

import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcdecode.dsl import parse_grammar, desugar
from gcdecode.automata import compile_grammar
from gcdecode.pda import PDA
from gcdecode.vocabulary import Vocabulary
from gcdecode.scorer import SeededScorer
from gcdecode.decoding import ConstrainedDecoder
from gcdecode.randgen import random_vocab_pieces
from gcdecode.report import render_html_report, ReportData

# 冰壶比赛记录:局数、每壶投掷方、落点区域、得分
CURLING_GRAMMAR = """
# 比赛记录:若干局,每局若干壶
record ::= "{" "frames" ":" farr "," "winner" ":" side "}";
farr   ::= "[" frame ("," frame){0,1} "]";
frame  ::= "{" "no" ":" #int(1, 4) "," "shots" ":" sarr "}";
sarr   ::= "[" shot ("," shot){0,2} "]";
shot   ::= "{" "team" ":" side "," "zone" ":" zone ","
           "pts" ":" #int(0, 5) "}";
side   ::= #enum("RED" | "YEL");
zone   ::= #enum("HOUSE" | "RING" | "GUARD" | "OUT");
"""


def main(open_browser: bool = False) -> str:
    rng = random.Random(20261122)
    chars = "{}[],:\"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pieces = random_vocab_pieces(rng, 110, chars)

    g_raw = parse_grammar(CURLING_GRAMMAR)
    g = desugar(g_raw)
    cg = compile_grammar(g)
    pda = PDA(cg)
    vocab = Vocabulary(pieces, eos_piece="<eos>")
    scorer = SeededScorer(vocab.size, seed=42)
    dec = ConstrainedDecoder(pda, vocab, scorer)
    # 给 EOS 一个适度先验偏置:可接受时结束符更有竞争力,
    # 模拟真实模型学到的"写完即停",避免演示在无界结构里空转。
    eos_bias = 3.0
    scorer.bias[vocab.eos_index] += eos_bias

    # 三种约束解码
    t0 = time.perf_counter_ns()
    greedy = dec.decode("greedy", max_steps=400)
    sample = dec.decode("sample", max_steps=400, temperature=0.8)
    beam = dec.beam_search(beam_width=4, max_steps=400)
    # 无约束基线(不施加 EOS 偏置,对照原始分布)
    scorer.bias[vocab.eos_index] -= eos_bias
    unconstrained = dec.decode("greedy", constrained=False, max_steps=120)
    scorer.bias[vocab.eos_index] += eos_bias
    t1 = time.perf_counter_ns()

    def trace_to_json(tr, include_steps=True):
        return {
            "mode": tr.mode,
            "constrained": tr.constrained,
            "accepted": tr.accepted,
            "final_text": tr.final_text,
            "token_ids": tr.token_ids,
            "total_ns": tr.total_ns,
            "total_mask_ns": tr.total_mask_ns,
            "total_score_ns": tr.total_score_ns,
            "mask_cache_hits": tr.mask_cache_hits,
            "mask_cache_misses": tr.mask_cache_misses,
            "flat_mask_uses": tr.flat_mask_uses,
            "eos_piece": vocab.pieces[vocab.eos_index],
            "pieces": vocab.pieces,
            "steps": ([_step_json(s, vocab) for s in tr.steps]
                      if include_steps else []),
            "beam_snapshots": [
                {"step": snap.step,
                 "beams": snap.beams} for snap in tr.beam_snapshots],
        }

    traces = {
        "贪心(约束)": trace_to_json(greedy),
        "温度采样(约束)": trace_to_json(sample),
        "束搜索 beam=4(约束)": trace_to_json(beam, include_steps=False),
    }

    # 束搜索没有逐步明细,用贪心步骤供页面展示;同时单独保留其快照
    overhead = ((greedy.total_ns - unconstrained.total_ns)
                / max(unconstrained.total_ns, 1) * 100.0)
    stats = {
        "unconstrained_total_ns": unconstrained.total_ns,
        "overhead_pct": round(overhead, 1),
        "num_machine_states": pda.num_machine_states,
        "num_flat_masks": len(dec.flat_masks),
        "wall_ns": t1 - t0,
    }

    desugared = "\n".join(
        f"{name} ::= {_expr_str(g.rules[name])};"
        for name in g.rule_names())

    data = ReportData(
        grammar_text=CURLING_GRAMMAR.strip(),
        desugared_text=desugared,
        traces=traces,
        stats=stats,
    )
    html_text = render_html_report(data)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "curling.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    # 控制台摘要
    print("=" * 68)
    print("冰壶比赛记录 · 文法约束解码演示")
    print("=" * 68)
    for label, tr in [("贪心", greedy), ("采样", sample), ("束搜索", beam)]:
        flag = "✓ 被文法接受" if tr.accepted else "✗ 未接受"
        print(f"[{label}] {flag}  耗时 {tr.total_ns/1000:.1f} µs"
              f"  掩码 {tr.total_mask_ns/1000:.1f} µs")
        print(f"        {tr.final_text[:90]}")
    print(f"[无约束基线] 耗时 {unconstrained.total_ns/1000:.1f} µs"
          f"  接受={unconstrained.accepted}")
    print(f"约束相对开销 {stats['overhead_pct']}% · "
          f"机器状态 {pda.num_machine_states} · "
          f"平坦掩码 {len(dec.flat_masks)}")
    print(f"报告已写入: {out_path}")
    if open_browser:
        import webbrowser
        webbrowser.open("file://" + out_path)
    return out_path


def _step_json(step, vocab):
    # 只保留页面展示所需字段;完整 raw_logits 体积过大且页面不使用,
    # 被屏蔽词的最高分信息已由 top_blocked 携带。
    top_blocked = None
    if step.top_blocked:
        tid, piece, logit = step.top_blocked
        top_blocked = {"id": tid, "piece": piece, "logit": round(logit, 4)}
    return {
        "step": step.step,
        "prefix": step.prefix,
        "state_id": step.state_id,
        "allowed_ids": step.allowed_ids,
        "blocked_ids": step.blocked_ids,
        "eos_allowed": step.eos_allowed,
        "chosen_id": step.chosen_id,
        "chosen_piece": step.chosen_piece,
        "mask_ns": step.mask_ns,
        "score_ns": step.score_ns,
        "total_ns": step.total_ns,
        "top_blocked": top_blocked,
    }


def _expr_str(expr, depth: int = 0) -> str:
    from gcdecode.dsl import Lit, CharClass, Ref, Seq, Alt, Repeat
    if isinstance(expr, Lit):
        return f'"{expr.text}"'
    if isinstance(expr, CharClass):
        body = ",".join(f"{lo}-{hi}" if lo != hi else lo
                        for lo, hi in expr.ranges)
        return f"[^{body}]" if expr.negated else f"[{body}]"
    if isinstance(expr, Ref):
        return expr.name
    if isinstance(expr, Seq):
        return " ".join(_expr_str(e, depth + 1) for e in expr.items)
    if isinstance(expr, Alt):
        return " | ".join(_expr_str(e, depth + 1) for e in expr.items)
    if isinstance(expr, Repeat):
        inner = _expr_str(expr.child, depth + 1)
        if expr.min == 0 and expr.max == 1:
            return f"{inner}?"
        if expr.min == 0 and expr.max is None:
            return f"{inner}*"
        if expr.min == 1 and expr.max is None:
            return f"{inner}+"
        hi = "∞" if expr.max is None else expr.max
        return f"{inner}{{{expr.min},{hi}}}"
    return str(expr)


if __name__ == "__main__":
    main(open_browser="--open" in sys.argv)
