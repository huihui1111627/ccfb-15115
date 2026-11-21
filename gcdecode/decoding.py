"""约束解码:掩码计算 + 贪心 / 温度采样 / 束搜索。

掩码计算流程
------------
1. 词表挂在 trie 上。给定当前机器状态,沿 trie 逐字符推进 PDA:
   - 走到某 trie 节点的终止词项,且当前机器状态仍有活性,
     该词项可用;
   - 某条边把 PDA 推进到 ``None``(必死),该子树整棵剪掉。
   这样一个跨多个文法符号的词项、或只停在符号中间的词项都能正确判定。
2. 对"无调用边的平坦规则 + 空栈"机器状态,启动时按 DFA 状态
   预计算整座位图;带栈状态在线 trie 计算,结果按机器状态编号缓存,
   trie 子问题按 (机器状态, trie 节点) 记忆化。
3. 结束符(EOS)仅当机器状态处于接受配置时可用 —— 文法没写完
   不可能出结束符。
4. 被屏蔽词项的 logit 置为负无穷。

束搜索中每条束各自携带独立的 scorer 隐状态和 PDA 机器状态。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .pda import PDA, MachineState
from .vocabulary import Vocabulary, TrieNode
from .scorer import SeededScorer, _Rng

NEG_INF = float("-inf")

__all__ = ["ConstrainedDecoder", "DecodeStep", "DecodeTrace", "BeamSnapshot"]


@dataclass
class DecodeStep:
    step: int
    prefix: str
    state_id: int
    allowed_ids: list[int]
    blocked_ids: list[int]
    eos_allowed: bool
    chosen_id: int
    chosen_piece: str
    raw_logits: list[float]
    adjusted_logits: list[float]
    mask_ns: int          # 约束掩码计算耗时
    score_ns: int         # 打分耗时
    total_ns: int         # 整步耗时
    top_blocked: Optional[tuple[int, str, float]] = None


@dataclass
class BeamSnapshot:
    step: int
    beams: list[dict]                      # tokens/text/score/state_id/finished


@dataclass
class DecodeTrace:
    mode: str
    constrained: bool
    steps: list[DecodeStep] = field(default_factory=list)
    beam_snapshots: list[BeamSnapshot] = field(default_factory=list)
    final_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    accepted: bool = False
    total_mask_ns: int = 0
    total_score_ns: int = 0
    total_ns: int = 0
    mask_cache_hits: int = 0
    mask_cache_misses: int = 0
    flat_mask_uses: int = 0


class ConstrainedDecoder:
    def __init__(self, pda: PDA, vocab: Vocabulary, scorer: SeededScorer,
                 sample_seed: int = 777):
        self.pda = pda
        self.vocab = vocab
        self.scorer = scorer
        self._sample_rng = _Rng(sample_seed)
        # trie 节点稳定编号(记忆化用)
        self._node_ids: dict[int, int] = {}
        self._assign_node_ids()
        # 平坦规则(全规则无调用边)的 DFA 状态 -> 预计算位图
        self.flat_masks: dict[tuple[int, int], int] = {}
        self.flat_rules: set[int] = set()
        # 在线 trie 子问题记忆化:(机器状态 id, trie 节点 id) -> 子树位图
        self._trie_memo: dict[tuple[int, int], int] = {}
        self._precompute_flat_masks()

    # ------------------------------------------------------------------ #
    # 预计算
    # ------------------------------------------------------------------ #
    def _assign_node_ids(self) -> None:
        nid = 0
        stack = [self.vocab.root]
        while stack:
            node = stack.pop()
            self._node_ids[id(node)] = nid
            nid += 1
            stack.extend(node.children.values())

    def _flat_rule_ids(self) -> set[int]:
        flat: set[int] = set()
        for r, rule in enumerate(self.pda.rules):
            has_call = any(rule.dfa_calls[s] for s in range(rule.num_states))
            if not has_call:
                flat.add(r)
        return flat

    def _precompute_flat_masks(self) -> None:
        """对无调用边规则的每个 DFA 状态,离线跑完整棵 trie 得位图。"""
        self.flat_rules = self._flat_rule_ids()
        for r in self.flat_rules:
            rule = self.pda.rules[r]
            for s in range(rule.num_states):
                seed_configs = frozenset({(r, s, ())})
                ms = self.pda._make_state(seed_configs)
                mask = self._trie_collect(ms, self.vocab.root,
                                          allow_memo=False)
                self.flat_masks[(r, s)] = mask

    # ------------------------------------------------------------------ #
    # trie 收集
    # ------------------------------------------------------------------ #
    def _trie_collect(self, ms: MachineState, node: TrieNode,
                      allow_memo: bool = True) -> int:
        if allow_memo:
            key = (ms.state_id, self._node_ids[id(node)])
            cached = self._trie_memo.get(key)
            if cached is not None:
                return cached
        mask = 0
        for tid in node.terminal:
            # 终止词项:当前机器状态可续写(前缀活性)或已可接受(EOS)。
            # 接受态无字符出边、前缀活性为假,但词项本身已构成完整合法串,
            # 必须允许(之后由 EOS 收尾)。
            if self.pda.machine_state_viable(ms):
                mask |= self.vocab.bit[tid]
        for ch, child in node.children.items():
            nxt = self.pda.advance_char(ms, ch)
            if nxt is not None:
                mask |= self._trie_collect(nxt, child)
        if allow_memo:
            self._trie_memo[(ms.state_id, self._node_ids[id(node)])] = mask
        return mask

    # ------------------------------------------------------------------ #
    # 对外掩码接口
    # ------------------------------------------------------------------ #
    def allowed_mask(self, ms: MachineState) -> tuple[int, dict]:
        """返回 (位图, 计时/缓存统计)。命中平坦预计算则零在线开销。"""
        t0 = time.perf_counter_ns()
        cached = self.pda.mask_cache.get(ms.state_id)
        stats = {"cache_hit": cached is not None, "flat": False}
        if cached is not None:
            mask = cached
        elif len(ms.configs) == 1:
            (r, s, stk), = ms.configs
            if not stk and r in self.flat_rules:
                mask = self.flat_masks[(r, s)]
                stats["flat"] = True
                self.pda.mask_cache[ms.state_id] = mask
            else:
                mask = self._trie_collect(ms, self.vocab.root)
                self.pda.mask_cache[ms.state_id] = mask
        else:
            mask = self._trie_collect(ms, self.vocab.root)
            self.pda.mask_cache[ms.state_id] = mask
        if ms.accepting:
            if self.vocab.eos_index is not None:
                mask |= self.vocab.bit[self.vocab.eos_index]
        else:
            if self.vocab.eos_index is not None:
                mask &= ~self.vocab.bit[self.vocab.eos_index]
        t1 = time.perf_counter_ns()
        stats["ns"] = t1 - t0
        return mask, stats

    # ------------------------------------------------------------------ #
    # logit 调整
    # ------------------------------------------------------------------ #
    def apply_mask(self, logits: list[float], mask: int) -> list[float]:
        out = list(logits)
        for i in range(len(out)):
            if not (mask & (1 << i)):
                out[i] = NEG_INF
        return out

    # ------------------------------------------------------------------ #
    # 选择策略
    # ------------------------------------------------------------------ #
    def _greedy_pick(self, adjusted: list[float]) -> int:
        best, best_id = NEG_INF, -1
        for i, v in enumerate(adjusted):
            if v > best:
                best, best_id = v, i
        if best_id < 0:
            raise RuntimeError("没有任何可用词项(文法与词表不兼容?)")
        return best_id

    def _sample_pick(self, adjusted: list[float], temperature: float) -> int:
        if temperature <= 0:
            return self._greedy_pick(adjusted)
        exps: list[tuple[int, float]] = []
        m = max(v for v in adjusted if v != NEG_INF)
        z = 0.0
        for i, v in enumerate(adjusted):
            if v == NEG_INF:
                continue
            w = math.exp((v - m) / temperature)
            exps.append((i, w))
            z += w
        u = (self._sample_rng.next_u32() + 1) / 4294967297.0 * z
        acc = 0.0
        for i, w in exps:
            acc += w
            if u <= acc:
                return i
        return exps[-1][0]

    # ------------------------------------------------------------------ #
    # 单束解码(贪心 / 采样)
    # ------------------------------------------------------------------ #
    def decode(self, mode: str = "greedy", *, constrained: bool = True,
               max_steps: int = 128, temperature: float = 1.0,
               fixed_steps: Optional[int] = None) -> DecodeTrace:
        assert mode in ("greedy", "sample")
        trace = DecodeTrace(mode=mode, constrained=constrained)
        ms: Optional[MachineState] = self.pda.initial
        scorer_state = self.scorer.initial_state()
        prefix = ""
        t_start = time.perf_counter_ns()
        step = 0
        while step < max_steps:
            if fixed_steps is not None and step >= fixed_steps:
                break
            step_t0 = time.perf_counter_ns()
            ts0 = time.perf_counter_ns()
            scored = self.scorer.score(scorer_state)
            logits = scored.logits
            ts1 = time.perf_counter_ns()

            if constrained:
                assert ms is not None
                mask, mstats = self.allowed_mask(ms)
                if mstats["cache_hit"]:
                    trace.mask_cache_hits += 1
                else:
                    trace.mask_cache_misses += 1
                if mstats.get("flat"):
                    trace.flat_mask_uses += 1
                adjusted = self.apply_mask(logits, mask)
                mask_ns = mstats["ns"]
                allowed = self.vocab.decode_mask(
                    mask & ~(self.vocab.bit[self.vocab.eos_index]
                             if self.vocab.eos_index is not None else 0))
                blocked = [i for i in range(self.vocab.size)
                           if not (mask & (1 << i))]
                eos_allowed = (self.vocab.eos_index is not None
                               and bool(mask & self.vocab.bit[
                                   self.vocab.eos_index]))
            else:
                adjusted = list(logits)
                mask_ns = 0
                mask = self.vocab.all_bits
                allowed = self.vocab.decode_mask(mask)
                blocked = []
                eos_allowed = True

            if mode == "greedy":
                chosen = self._greedy_pick(adjusted)
            else:
                chosen = self._sample_pick(adjusted, temperature)

            if chosen == self.vocab.eos_index:
                step_t1 = time.perf_counter_ns()
                trace.steps.append(self._make_step(
                    step, prefix, ms, allowed, blocked, eos_allowed,
                    chosen, logits, adjusted, mask_ns, ts1 - ts0,
                    step_t1 - step_t0))
                trace.total_mask_ns += mask_ns
                trace.total_score_ns += ts1 - ts0
                break

            piece = self.vocab.pieces[chosen]
            if constrained:
                assert ms is not None
                nxt = self.pda.advance_char(ms, piece[0])
                for ch in piece[1:]:
                    if nxt is None:
                        break
                    nxt = self.pda.advance_char(nxt, ch)
                if nxt is None:
                    raise RuntimeError(
                        f"约束实现错误:词项 {piece!r} 通过掩码但 PDA 拒绝")
                ms = nxt
            scorer_state = self.scorer.step_state(scorer_state, chosen)
            prefix += piece

            step_t1 = time.perf_counter_ns()
            trace.steps.append(self._make_step(
                step, prefix, ms, allowed, blocked, eos_allowed,
                chosen, logits, adjusted, mask_ns, ts1 - ts0,
                step_t1 - step_t0))
            trace.total_mask_ns += mask_ns
            trace.total_score_ns += ts1 - ts0
            step += 1

        t_end = time.perf_counter_ns()
        trace.total_ns = t_end - t_start
        trace.final_text = prefix
        trace.token_ids = [s.chosen_id for s in trace.steps]
        trace.accepted = self.pda.accepts(prefix)
        return trace

    def _make_step(self, step, prefix, ms, allowed, blocked, eos_allowed,
                   chosen, logits, adjusted, mask_ns, score_ns,
                   total_ns) -> DecodeStep:
        top_blocked = None
        if blocked:
            best = max(blocked, key=lambda i: logits[i])
            top_blocked = (best, self.vocab.pieces[best], logits[best])
        return DecodeStep(
            step=step, prefix=prefix,
            state_id=ms.state_id if ms is not None else -1,
            allowed_ids=allowed, blocked_ids=blocked,
            eos_allowed=eos_allowed, chosen_id=chosen,
            chosen_piece=self.vocab.pieces[chosen],
            raw_logits=list(logits), adjusted_logits=adjusted,
            mask_ns=mask_ns, score_ns=score_ns, total_ns=total_ns,
            top_blocked=top_blocked)

    # ------------------------------------------------------------------ #
    # 束搜索(每束独立机器状态与 scorer 状态)
    # ------------------------------------------------------------------ #
    def beam_search(self, beam_width: int = 4, *, max_steps: int = 128,
                    constrained: bool = True) -> DecodeTrace:
        trace = DecodeTrace(mode=f"beam{beam_width}", constrained=constrained)
        t_start = time.perf_counter_ns()

        @dataclass
        class Beam:
            tokens: list[int]
            text: str
            score: float
            scorer_state: list[float]
            ms: Optional[MachineState]
            finished: bool = False

        beams = [Beam(tokens=[], text="", score=0.0,
                      scorer_state=self.scorer.initial_state(),
                      ms=self.pda.initial if constrained else None)]
        step = 0
        while step < max_steps and not all(b.finished for b in beams):
            candidates: list[Beam] = []
            for b in beams:
                if b.finished:
                    candidates.append(b)
                    continue
                step_t0 = time.perf_counter_ns()
                ts0 = time.perf_counter_ns()
                logits = self.scorer.score(b.scorer_state).logits
                ts1 = time.perf_counter_ns()
                if constrained:
                    mask, mstats = self.allowed_mask(b.ms)
                    if mstats["cache_hit"]:
                        trace.mask_cache_hits += 1
                    else:
                        trace.mask_cache_misses += 1
                    if mstats.get("flat"):
                        trace.flat_mask_uses += 1
                    adjusted = self.apply_mask(logits, mask)
                    trace.total_mask_ns += mstats["ns"]
                else:
                    adjusted = list(logits)
                    mask = self.vocab.all_bits
                    trace.total_mask_ns += 0
                trace.total_score_ns += ts1 - ts0
                # 每条束取前 beam_width 个候选
                scored = [(i, v) for i, v in enumerate(adjusted)
                          if v != NEG_INF]
                scored.sort(key=lambda x: x[1], reverse=True)
                for tid, val in scored[:beam_width]:
                    logp = val - math.log(sum(
                        math.exp(w - max(x[1] for x in scored))
                        for _, w in scored))
                    if tid == self.vocab.eos_index:
                        if constrained and not b.ms.accepting:
                            continue
                        candidates.append(Beam(
                            tokens=b.tokens + [tid], text=b.text,
                            score=b.score + logp,
                            scorer_state=b.scorer_state, ms=b.ms,
                            finished=True))
                        continue
                    piece = self.vocab.pieces[tid]
                    if constrained:
                        nxt = self.pda.advance_char(b.ms, piece[0])
                        for ch in piece[1:]:
                            if nxt is None:
                                break
                            nxt = self.pda.advance_char(nxt, ch)
                        if nxt is None:
                            continue
                    else:
                        nxt = None
                    candidates.append(Beam(
                        tokens=b.tokens + [tid], text=b.text + piece,
                        score=b.score + logp,
                        scorer_state=self.scorer.step_state(
                            b.scorer_state, tid),
                        ms=nxt if constrained else None))
                _ = step_t0
            # 长度归一化(Google GNMT 风格,alpha=0.7)后剪枝,
            # 避免长束仅因累加项多而永远压过已结束的短束
            def norm(b: "Beam") -> float:
                length = len(b.tokens)
                lp = ((5 + length) ** 0.7) / (6 ** 0.7)
                return b.score / lp

            candidates.sort(key=norm, reverse=True)
            beams = candidates[:beam_width]
            trace.beam_snapshots.append(BeamSnapshot(
                step=step,
                beams=[{"tokens": list(b.tokens), "text": b.text,
                        "score": b.score, "norm_score": norm(b),
                        "state_id": b.ms.state_id if b.ms else -1,
                        "finished": b.finished} for b in beams]))
            step += 1

        best = max(beams, key=lambda b: b.score / (
            ((5 + len(b.tokens)) ** 0.7) / (6 ** 0.7)))
        t_end = time.perf_counter_ns()
        trace.total_ns = t_end - t_start
        trace.final_text = best.text
        trace.token_ids = [t for t in best.tokens
                           if t != self.vocab.eos_index]
        trace.accepted = self.pda.accepts(best.text)
        # 兼容页面字段
        trace.steps = []
        return trace
