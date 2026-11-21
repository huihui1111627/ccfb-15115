import sys
sys.path.insert(0, ".")

from gcdecode.dsl import parse_grammar, desugar
from gcdecode.automata import compile_grammar
from gcdecode.pda import PDA
from gcdecode.vocabulary import Vocabulary
from gcdecode.scorer import SeededScorer
from gcdecode.decoding import ConstrainedDecoder

text = """
record ::= obj;
obj    ::= "{" pair ("," pair){0,3} "}";
pair   ::= "k" ":" [a-z]{1,3};
"""
g = desugar(parse_grammar(text))
pda = PDA(compile_grammar(g))

pieces = ["<eos>", "{", "}", ",", "k", ":", "a", "b", "c", "x", "y",
          "ab", "abc", "k:", "{k", ":a", "xy", "zz", "k:a", "{}", ",k"]
vocab = Vocabulary(pieces, eos_piece="<eos>")
scorer = SeededScorer(vocab.size, seed=42)
dec = ConstrainedDecoder(pda, vocab, scorer)

for mode in ("greedy", "sample"):
    tr = dec.decode(mode, constrained=True, max_steps=200, temperature=0.8)
    print(f"[{mode}] accepted={tr.accepted} text={tr.final_text!r} "
          f"steps={len(tr.steps)} mask_total_us={tr.total_mask_ns/1000:.1f}")
    assert tr.accepted, mode
    # 每一步选用的词项当时都在掩码内
    for st in tr.steps:
        assert st.chosen_id in st.allowed_ids or st.chosen_id == vocab.eos_index

tr = dec.beam_search(beam_width=4, max_steps=40)
print(f"[beam4] text={tr.final_text!r} snapshots={len(tr.beam_snapshots)}")
# 束搜索:所有束的文本必须是合法前缀(约束不能放过错词)
for snap in tr.beam_snapshots:
    for b in snap.beams:
        ms = pda.consume(b["text"])
        assert ms is not None, f"束文本非法: {b['text']!r}"
        if b["finished"]:
            assert pda.accepts(b["text"])
# 强制验证:任何接受态下 EOS 一定可用(没写完时一定不可用)
acc_ms = pda.consume("{k:ab}")
assert acc_ms.accepting
acc_mask, _ = dec.allowed_mask(acc_ms)
assert vocab.is_allowed(acc_mask, vocab.eos_index)

# 无约束对照
tr_unc = dec.decode("greedy", constrained=False, max_steps=30)
print(f"[unconstrained] accepted={tr_unc.accepted} "
      f"text={tr_unc.final_text!r}")

# 第一步掩码内容检查
ms = pda.initial
mask, stats = dec.allowed_mask(ms)
print("first-step allowed:", vocab.mask_pieces(mask))
print("eos allowed at start:", vocab.is_allowed(mask, vocab.eos_index))
assert not vocab.is_allowed(mask, vocab.eos_index), "文法没写完不能出 EOS"
print("stats:", stats)
print("flat masks:", len(dec.flat_masks), "machine states:", pda.num_machine_states)
print("OK")
