"""核心正确性测试:约束引擎 vs 独立暴力实现。

验收项(对应需求)
------------------
1. 几十种随机文法 × 数千个前缀,每一步可用词集合与"逐词项拼接后
   交给文法暴力判定"的结果完全一致;
2. 贪心 / 采样 / 束搜索产出的所有结果都能被文法完整接受;
3. 最小化前后的 DFA 接受同一语言;
4. 文法未写完时 EOS 永不开放,接受态下 EOS 必开放;
5. 跨符号词项、停在符号中间的词项判定正确。
"""

import random
import unittest

from gcdecode.dsl import parse_grammar, desugar
from gcdecode.automata import (compile_grammar, _NFABuilder, _build_dfa,
                               _minimize, Alphabet, GrammarCompileError)
from gcdecode.pda import PDA
from gcdecode.vocabulary import Vocabulary
from gcdecode.scorer import SeededScorer
from gcdecode.decoding import ConstrainedDecoder
from gcdecode.nfa_interp import BruteForceInterpreter, brute_force_mask
from gcdecode.randgen import (random_grammar, random_prefixes,
                              random_vocab_pieces)

# 几十条文法 × 每条约百个前缀,累计数千前缀、数万次词项判定
N_GRAMMARS = 40
PREFIXES_PER_GRAMMAR = 120
VOCAB_SIZE = 60
MAX_DERIVE_LEN = 16
MASTER_SEED = 20261122


def build_pipeline(text: str, pieces, seed: int = 1):
    g = desugar(parse_grammar(text))
    cg = compile_grammar(g)
    pda = PDA(cg)
    vocab = Vocabulary(pieces, eos_piece="<eos>")
    scorer = SeededScorer(vocab.size, seed=seed)
    dec = ConstrainedDecoder(pda, vocab, scorer)
    return g, cg, pda, vocab, dec


class TestMaskEquivalence(unittest.TestCase):
    def test_random_grammars_masks_match_bruteforce(self):
        rng = random.Random(MASTER_SEED)
        total_prefixes = 0
        total_tokens = 0
        skipped = 0
        for gi in range(N_GRAMMARS):
            g, text = random_grammar(rng)
            chars = sorted({c for e in g.rules.values()
                            for c in _collect_chars(e)})
            alpha_chars = "".join(chars) or "abxy"
            pieces = random_vocab_pieces(
                rng, VOCAB_SIZE, alpha_chars + "abxy")
            try:
                _g, cg, pda, vocab, dec = build_pipeline(text, pieces,
                                                         seed=gi + 1)
            except GrammarCompileError:
                skipped += 1
                continue
            interp = BruteForceInterpreter(g, repeat_cap=4)
            prefixes = random_prefixes(rng, g, PREFIXES_PER_GRAMMAR,
                                       max_len=12)
            for prefix in prefixes:
                total_prefixes += 1
                bf_mask = brute_force_mask(interp, vocab, prefix,
                                           MAX_DERIVE_LEN)
                ms = pda.consume(prefix)
                if ms is None:
                    self.assertFalse(
                        interp.can_continue(prefix),
                        msg=f"文法:\n{text}\n前缀 {prefix!r} 引擎判死但暴力存活")
                    continue
                engine_mask, _ = dec.allowed_mask(ms)
                for tid in range(vocab.size):
                    total_tokens += 1
                    bf_ok = bool(bf_mask & vocab.bit[tid])
                    eng_ok = bool(engine_mask & vocab.bit[tid])
                    self.assertEqual(
                        bf_ok, eng_ok,
                        msg=(f"文法:\n{text}\n前缀 {prefix!r} 词项 "
                             f"{vocab.pieces[tid]!r}: 暴力={bf_ok} "
                             f"引擎={eng_ok}"))
        print(f"\n[掩码对照] 文法={N_GRAMMARS}(跳过左递归 {skipped}) "
              f"前缀={total_prefixes} 词项判定={total_tokens}")
        self.assertGreaterEqual(total_prefixes, 2000)
        self.assertGreaterEqual(total_tokens, 50000)

    def test_handwritten_nested_grammar(self):
        text = """
        root ::= arr;
        arr    ::= "[" item ("," item)* "]";
        item   ::= obj | num;
        obj    ::= "{" kv ("," kv)* "}";
        kv     ::= "k" ":" num;
        num    ::= #int(0, 15);
        """
        rng = random.Random(7)
        chars = "[]{},:k0123456789"
        pieces = random_vocab_pieces(rng, 80, chars)
        g, cg, pda, vocab, dec = build_pipeline(text, pieces)
        interp = BruteForceInterpreter(g, repeat_cap=4)
        prefixes = random_prefixes(rng, g, 400, max_len=14)
        checked = 0
        for prefix in prefixes:
            bf_mask = brute_force_mask(interp, vocab, prefix, 16)
            ms = pda.consume(prefix)
            if ms is None:
                self.assertFalse(interp.can_continue(prefix), prefix)
                continue
            engine_mask, _ = dec.allowed_mask(ms)
            for tid in range(vocab.size):
                checked += 1
                self.assertEqual(
                    bool(bf_mask & vocab.bit[tid]),
                    bool(engine_mask & vocab.bit[tid]),
                    msg=f"{prefix!r} / {vocab.pieces[tid]!r}")
        print(f"[嵌套文法] 词项判定={checked}")
        self.assertGreater(checked, 5000)


class TestDFAMinimization(unittest.TestCase):
    def test_minimized_preserves_language(self):
        rng = random.Random(99)
        checked_strings = 0
        # 规模严格受控:15 条文法,每条只用少量枚举前缀 + 固定随机串,
        # 避免递归/嵌套重复文法上的枚举爆炸,保证测试快速确定。
        for _ in range(15):
            g, _text = random_grammar(rng)
            rb = {n: i for i, n in enumerate(g.rule_names())}
            alphabet = Alphabet()
            builders = []
            for name in g.rule_names():
                b = _NFABuilder(alphabet, rb)
                frag = b.build(g.rules[name])
                b.states[frag.end].final = True
                builders.append(b)
            # 只用有界枚举的前若干个代表串 + 一批随机串
            interp = BruteForceInterpreter(g, repeat_cap=2)
            candidates = set()
            for s in interp.enumerate_prefixes(6):
                candidates.add(s)
                if len(candidates) >= 60:
                    break
            for _ in range(60):
                candidates.add("".join(
                    rng.choice("abxy0123456789{}\",:")
                    for _ in range(rng.randint(0, 8))))
            for b in builders:
                raw, index = _build_dfa(b, alphabet.num_cats)
                part_id, _new_n = _minimize(raw, alphabet.num_cats)
                # 最小化转移表与终态只构造一次,避免逐串重复计算
                min_trans: dict[int, dict[int, int]] = {}
                for old_id, st in enumerate(raw):
                    pid = part_id[old_id]
                    slot = min_trans.setdefault(pid, {})
                    for cat, tgt in st.trans.items():
                        slot[cat] = part_id[tgt]
                min_finals = {part_id[i] for i, st in enumerate(raw)
                              if st.final}
                min_start = part_id[0]

                def run_raw(s):
                    sid = 0
                    for ch in s:
                        cat = alphabet.char_cat.get(ch, alphabet.num_cats)
                        nxt = raw[sid].trans.get(cat)
                        if nxt is None:
                            return False
                        sid = nxt
                    return raw[sid].final

                def run_min(s):
                    sid = min_start
                    for ch in s:
                        cat = alphabet.char_cat.get(ch, alphabet.num_cats)
                        nxt = min_trans.get(sid, {}).get(cat)
                        if nxt is None:
                            return False
                        sid = nxt
                    return sid in min_finals

                for s in candidates:
                    checked_strings += 1
                    self.assertEqual(run_raw(s), run_min(s),
                                     msg=f"串 {s!r} 最小化改变接受性")
        print(f"[最小化语言等价] 串判定={checked_strings}")
        self.assertGreater(checked_strings, 1500)

    def test_minimized_reduces_or_keeps_states(self):
        # 两条独立但同构的分支:子集构造产生冗余状态,最小化应能合并。
        # 用足够大的等价结构,确保 pre_min 严格大于最小化结果。
        text = 'r ::= ("a" "b" "c") | ("a" "b" "c") | "x";'
        g = desugar(parse_grammar(text))
        cg = compile_grammar(g)
        rule = cg.rules[0]
        self.assertLessEqual(rule.num_states, rule.pre_min_states)
        self.assertLess(rule.num_states, rule.pre_min_states)

    def test_int_range_branches_not_merged(self):
        # 回归:0-3 与 0-9 两个范围分支不能被最小化错误合并
        text = 'r ::= {"f0": #int(0, 3)} | {"f0": #int(0, 9)};'
        g = desugar(parse_grammar(text))
        cg = compile_grammar(g)
        pda = PDA(cg)
        self.assertTrue(pda.accepts('{"f0":4}'))   # 仅第二分支
        self.assertTrue(pda.accepts('{"f0":3}'))   # 两个分支
        self.assertTrue(pda.accepts('{"f0":9}'))   # 仅第二分支
        self.assertFalse(pda.accepts('{"f0":10}'))


def _run_partition(raw, part_id, alphabet, s):
    sid = part_id[0]
    # 重建最小化转移
    trans: dict[int, dict[int, int]] = {}
    for old_id, st in enumerate(raw):
        pid = part_id[old_id]
        trans.setdefault(pid, {})
        for cat, t in st.trans.items():
            trans[pid][cat] = part_id[t]
    finals = {part_id[i] for i, st in enumerate(raw) if st.final}
    for ch in s:
        cat = alphabet.char_cat.get(ch, alphabet.num_cats)
        nxt = trans.get(sid, {}).get(cat)
        if nxt is None:
            return False
        sid = nxt
    return sid in finals


class TestDecodingAcceptance(unittest.TestCase):
    def setUp(self):
        text = """
        root ::= frame;
        frame ::= "{" shot ("," shot){0,4} "}";
        shot  ::= "s" ":" side ":" #int(0, 4);
        side  ::= #enum("R" | "Y");
        """
        rng = random.Random(3)
        pieces = random_vocab_pieces(rng, 90, "{},:sRY0123456789")
        self.g, self.cg, self.pda, self.vocab, self.dec = \
            build_pipeline(text, pieces, seed=11)

    def test_greedy_output_accepted(self):
        for seed in range(5):
            dec = ConstrainedDecoder(
                self.pda, self.vocab,
                SeededScorer(self.vocab.size, seed=100 + seed))
            tr = dec.decode("greedy", max_steps=120)
            self.assertTrue(tr.accepted, tr.final_text)

    def test_sample_outputs_accepted(self):
        for seed in range(5):
            dec = ConstrainedDecoder(
                self.pda, self.vocab,
                SeededScorer(self.vocab.size, seed=200 + seed))
            tr = dec.decode("sample", max_steps=200, temperature=0.7)
            self.assertTrue(tr.accepted, tr.final_text)

    def test_beam_all_beams_valid(self):
        tr = self.dec.beam_search(beam_width=4, max_steps=80)
        for snap in tr.beam_snapshots:
            for b in snap.beams:
                ms = self.pda.consume(b["text"])
                self.assertIsNotNone(ms, b["text"])
                if b["finished"]:
                    self.assertTrue(self.pda.accepts(b["text"]))

    def test_eos_blocked_until_complete(self):
        seen = set()
        stack = [self.pda.initial]
        while stack and len(seen) < 200:
            ms = stack.pop()
            if ms.state_id in seen:
                continue
            seen.add(ms.state_id)
            mask, _ = self.dec.allowed_mask(ms)
            eos = self.vocab.eos_index
            if ms.accepting:
                self.assertTrue(mask & self.vocab.bit[eos])
            else:
                self.assertFalse(mask & self.vocab.bit[eos])
            for ch in "{},:sRY0123456789":
                nxt = self.pda.advance_char(ms, ch)
                if nxt is not None:
                    stack.append(nxt)

    def test_mask_never_empty_on_viable_prefix(self):
        rng = random.Random(5)
        from gcdecode.randgen import _sample_from_ast
        for _ in range(100):
            full = _sample_from_ast(rng, self.g.rules[self.g.start],
                                    self.g.rules, 12, [])
            cut = rng.randint(0, len(full))
            prefix = full[:cut]
            ms = self.pda.consume(prefix)
            self.assertIsNotNone(ms, prefix)
            mask, _ = self.dec.allowed_mask(ms)
            self.assertNotEqual(mask, 0, f"活前缀 {prefix!r} 掩码为空")


class TestCrossSymbolTokens(unittest.TestCase):
    """词项跨多个文法符号 / 停在符号中间(需求第 4 条)。"""

    def test_token_spanning_symbols(self):
        text = 'r ::= "ab" "cd";'
        g = desugar(parse_grammar(text))
        pieces = ["<eos>", "a", "ab", "abc", "abcd", "b", "cd", "x"]
        _g, cg, pda, vocab, dec = build_pipeline(text, pieces)
        mask, _ = dec.allowed_mask(pda.initial)
        for ok_piece in ("abcd", "abc", "ab", "a"):
            self.assertTrue(vocab.is_allowed(mask, vocab.pieces.index(ok_piece)),
                            ok_piece)
        self.assertFalse(vocab.is_allowed(mask, vocab.pieces.index("x")))
        ms2 = pda.consume("abcd")
        self.assertTrue(ms2.accepting)
        mask2, _ = dec.allowed_mask(ms2)
        self.assertEqual(vocab.decode_mask(mask2), [vocab.eos_index])

    def test_long_piece_partial_rejection(self):
        text = 'r ::= "abc";'
        pieces = ["<eos>", "ab", "abd", "abc", "abcd"]
        _g, cg, pda, vocab, dec = build_pipeline(text, pieces)
        mask, _ = dec.allowed_mask(pda.initial)
        self.assertTrue(vocab.is_allowed(mask, vocab.pieces.index("ab")))
        self.assertTrue(vocab.is_allowed(mask, vocab.pieces.index("abc")))
        self.assertFalse(vocab.is_allowed(mask, vocab.pieces.index("abd")))
        self.assertFalse(vocab.is_allowed(mask, vocab.pieces.index("abcd")))


class TestNegatedCharClass(unittest.TestCase):
    """取反字符类的通配语义(逐 NFA 状态歧义)。"""

    def test_negation_excludes_and_accepts_others(self):
        text = 'r ::= "2" [^z] [yb]{0,2};'
        g = desugar(parse_grammar(text))
        cg = compile_grammar(g)
        pda = PDA(cg)
        self.assertTrue(pda.accepts("2O"))
        self.assertTrue(pda.accepts("2a"))
        self.assertIsNone(pda.consume("2z"))
        self.assertTrue(pda.accepts("2yb"))

    def test_negation_ambiguity_with_parallel_enum(self):
        # 同一位置既有显式 enum 边又有通配边,两条路径都要保留
        text = 'r ::= #enum("OUT"){0,1} [^z];'
        g = desugar(parse_grammar(text))
        pda = PDA(compile_grammar(g))
        self.assertIsNotNone(pda.consume("O"))   # enum 路径
        self.assertIsNotNone(pda.consume("a"))   # 通配路径
        self.assertIsNone(pda.consume("z"))      # 通配排除 z


class TestLeftRecursionRejected(unittest.TestCase):
    def test_mandatory_cycle_rejected(self):
        text = "a ::= b;\nb ::= a;\n"
        with self.assertRaises(GrammarCompileError):
            compile_grammar(desugar(parse_grammar(text)))

    def test_tail_recursion_allowed(self):
        text = 'r ::= "x" r?;'
        cg = compile_grammar(desugar(parse_grammar(text)))
        pda = PDA(cg)
        self.assertTrue(pda.accepts("xxx"))
        self.assertTrue(pda.accepts("x"))
        self.assertFalse(pda.accepts(""))
        # 可选递归规则整体可空
        text2 = 'top ::= r?;\nr ::= "x" r?;'
        pda2 = PDA(compile_grammar(desugar(parse_grammar(text2))))
        self.assertTrue(pda2.accepts(""))


def _collect_chars(expr, out=None):
    from gcdecode.dsl import (Lit, CharClass, Ref, Seq, Alt, Repeat)
    if out is None:
        out = set()
    if isinstance(expr, Lit):
        out.update(expr.text)
    elif isinstance(expr, CharClass) and not expr.negated:
        for lo, hi in expr.ranges:
            for cp in range(ord(lo), ord(hi) + 1):
                out.add(chr(cp))
    elif isinstance(expr, Seq):
        for e in expr.items:
            _collect_chars(e, out)
    elif isinstance(expr, Alt):
        for e in expr.items:
            _collect_chars(e, out)
    elif isinstance(expr, Repeat):
        _collect_chars(expr.child, out)
    return out


if __name__ == "__main__":
    unittest.main(verbosity=2)
