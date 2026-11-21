"""独立暴力参考实现 —— 测试专用,与 NFA/DFA/PDA 代码路径完全独立。

直接在脱糖后的 AST 上做两件事:

1. ``match_endpos``:递归下降 + memo 匹配,返回字符串从位置 i 出发
   能匹配到的所有结束位置(多歧义文法一个输入可有多个结束点)。
2. ``enumerate_prefixes``:有界展开文法,枚举全部可派生串的前缀集合
   (无穷重复按 ``repeat_cap`` 截断)。掩码暴力判定即
   "词项接上后仍是某个可派生串的前缀",EOS 仅在完整接受时可用。

测试用它与 PDA/trie 路径逐前缀、逐词项比对,任何不一致都会暴露。
"""

from __future__ import annotations

from .dsl import (Grammar, Lit, CharClass, Ref, Seq, Alt, Repeat,
                  RangeExpr, expand_int_range)

__all__ = ["BruteForceInterpreter", "brute_force_mask"]


class BruteForceInterpreter:
    def __init__(self, grammar: Grammar, repeat_cap: int = 6):
        self.g = grammar
        self.cap = repeat_cap
        self.rule_expr = grammar.rules
        self.start = grammar.start
        # 显式 memo:(文本, 表达式对象 id, 位置) -> 结束位置集合。
        # 必须把文本放进键里:同一解释器会对很多不同串求值。
        self._memo: dict[tuple[str, int, int], frozenset[int]] = {}
        self._rule_memo: dict[tuple[str, str, int], frozenset[int]] = {}

    # ------------------------------------------------------------------ #
    # 递归下降匹配(返回所有可能的结束位置)
    # ------------------------------------------------------------------ #
    def match_endpos(self, text: str, rule: str, i: int) -> frozenset[int]:
        return self._match_rule(rule, text, i)

    def _depth_budget(self, text: str) -> int:
        # 一次完整匹配至多消费 len(text) 个字符;规则嵌套超过该深度后
        # 只剩不消费字符的递归环,不可能匹配成功,据此剪枝防栈溢出。
        return len(text) + 4

    def _match_expr(self, expr, text: str, i: int,
                    depth: int = 0,
                    call_stack: tuple[tuple[str, int], ...] = ()
                    ) -> frozenset[int]:
        if depth > self._depth_budget(text):
            return frozenset()
        key = (text, id(expr), i, depth)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        result = self._match_expr_uncached(expr, text, i, depth, call_stack)
        self._memo[key] = result
        return result

    def _match_expr_uncached(self, expr, text: str, i: int,
                             depth: int = 0,
                             call_stack: tuple[tuple[str, int], ...] = ()
                             ) -> frozenset[int]:
        if isinstance(expr, Lit):
            end = i + len(expr.text)
            return frozenset({end}) if text.startswith(expr.text, i) else frozenset()
        if isinstance(expr, CharClass):
            if i < len(text) and expr.match_char(text[i]):
                return frozenset({i + 1})
            return frozenset()
        if isinstance(expr, Ref):
            return self._match_rule(expr.name, text, i, depth + 1,
                                    call_stack)
        if isinstance(expr, Seq):
            ends = frozenset({i})
            for sub in expr.items:
                nxt: set[int] = set()
                for pos in ends:
                    nxt |= self._match_expr(sub, text, pos, depth,
                                            call_stack)
                ends = frozenset(nxt)
                if not ends:
                    break
            return ends
        if isinstance(expr, Alt):
            out: set[int] = set()
            for sub in expr.items:
                out |= self._match_expr(sub, text, i, depth, call_stack)
            return frozenset(out)
        if isinstance(expr, Repeat):
            return self._match_repeat(expr, text, i, depth, call_stack)
        if isinstance(expr, RangeExpr):
            return self._match_expr(
                expand_int_range(expr.lo, expr.hi), text, i, depth,
                call_stack)
        raise AssertionError(f"未处理的 AST 节点 {type(expr)}")

    def _match_rule(self, name: str, text: str, i: int,
                    depth: int = 0,
                    call_stack: tuple[tuple[str, int], ...] = ()
                    ) -> frozenset[int]:
        if depth > self._depth_budget(text):
            return frozenset()
        # 同一规则在同一文本位置重复进入:这是不消费字符的递归,
        # 不可能产生新的匹配位置,直接剪枝(左递归/深递归防护)。
        frame = (name, i)
        if frame in call_stack:
            return frozenset()
        key = (text, name, i, depth)
        cached = self._rule_memo.get(key)
        if cached is not None:
            return cached
        result = self._match_expr(
            self.rule_expr[name], text, i, depth, call_stack + (frame,))
        self._rule_memo[key] = result
        return result

    def _match_repeat(self, expr: Repeat, text: str, i: int,
                      depth: int = 0,
                      call_stack: tuple[tuple[str, int], ...] = ()
                      ) -> frozenset[int]:
        # 有界/无界重复统一按 cap 截断(测试文法的合理派生不会超过 cap)
        upper = self.cap if expr.max is None else min(expr.max, self.cap)
        # 深度预算:递归规则的重复各层都增加嵌套,超预算后无法在有限
        # 文本内完成匹配,直接剪枝,避免递归文法上的组合爆炸/栈溢出。
        layers_budget = max(0, self._depth_budget(text) - depth)
        upper = min(upper, layers_budget)
        # 按层 BFS:layer[k] = 恰好匹配 k 次后可达的位置集合
        layers: list[set[int]] = [set() for _ in range(upper + 1)]
        layers[0] = {i}
        for k in range(upper):
            nxt: set[int] = set()
            for pos in layers[k]:
                nxt |= self._match_expr(expr.child, text, pos, depth + 1,
                                         call_stack)
            layers[k + 1] = nxt
            if not layers[k + 1]:
                break
        out: set[int] = set()
        for k in range(expr.min, upper + 1):
            out |= layers[k]
        return frozenset(out)

    def initialize(self) -> "BruteForceInterpreter":
        # memo 在构造时初始化;保留方法以兼容旧调用
        return self

    # ------------------------------------------------------------------ #
    # 接受判定
    # ------------------------------------------------------------------ #
    def accepts(self, text: str) -> bool:
        return len(text) in self.match_endpos(text, self.start, 0)

    def is_prefix_viable(self, text: str, max_len: int | None = None) -> bool:
        """text 是否是某个可派生串的前缀(含完整接受)。

        用有界枚举得到的前缀集合判定 —— 与 brute_force_mask 完全同源,
        避免"匹配结束位置"在多歧义/部分消费场景下的口径不一致。
        """
        if max_len is None:
            max_len = max(len(text), self.cap) + 4
        return text in self.enumerate_prefixes(max_len)

    # ------------------------------------------------------------------ #
    # 有界枚举:全部可派生串 -> 前缀集合
    # ------------------------------------------------------------------ #
    def enumerate_prefixes(self, max_len: int) -> set[str]:
        prefixes: set[str] = set()
        for s in self._derive_rule(self.start, self.cap, max_len, ()):
            for k in range(len(s) + 1):
                prefixes.add(s[:k])
        return prefixes

    # 规则引用嵌套深度上限:无基例尾递归(如 r ::= "x" r)在枚举时
    # 不会终止,用较浅的深度上限截断,与 repeat_cap 共同保证枚举有界、
    # 规模可控(测试只需要代表性前缀,不需要完整语言)。
    REF_DEPTH_CAP = 3

    def _derive_expr(self, expr, cap: int, budget: int,
                     ref_stack: tuple[str, ...] = ()):
        if budget < 0:
            return
        if isinstance(expr, Lit):
            if len(expr.text) <= budget:
                yield expr.text
            return
        if isinstance(expr, CharClass):
            if budget >= 1:
                if expr.negated:
                    # 取反类:只枚举报文里实际可能出现的有限代表字符
                    candidates = ("a", "b", "x", "y", "0", "O", "R")
                    for c in candidates:
                        if expr.match_char(c):
                            yield c
                else:
                    for lo, hi in expr.ranges:
                        for cp in range(ord(lo), ord(hi) + 1):
                            yield chr(cp)
            return
        if isinstance(expr, Ref):
            if ref_stack.count(expr.name) >= self.REF_DEPTH_CAP:
                return
            yield from self._derive_rule(
                expr.name, cap, budget, ref_stack + (expr.name,))
            return
        if isinstance(expr, Seq):
            yield from self._derive_seq(expr.items, 0, "", cap, budget,
                                        ref_stack)
            return
        if isinstance(expr, Alt):
            for sub in expr.items:
                yield from self._derive_expr(sub, cap, budget, ref_stack)
            return
        if isinstance(expr, Repeat):
            upper = cap if expr.max is None else min(expr.max, cap)
            yield from self._derive_repeat(expr, cap, budget, upper,
                                           ref_stack)
            return
        if isinstance(expr, RangeExpr):
            yield from self._derive_expr(
                expand_int_range(expr.lo, expr.hi), cap, budget, ref_stack)
            return
        raise AssertionError(type(expr))

    def _derive_seq(self, items, idx, acc, cap, budget, ref_stack):
        if idx == len(items):
            yield acc
            return
        for tail in self._derive_expr(items[idx], cap, budget, ref_stack):
            if len(acc) + len(tail) <= budget:
                yield from self._derive_seq(items, idx + 1, acc + tail,
                                            cap, budget, ref_stack)

    def _derive_repeat(self, expr, cap, budget, upper, ref_stack):
        def layers(k, acc, rem_budget):
            if k >= expr.min:
                yield acc
            if k >= upper:
                return
            for piece in self._derive_expr(expr.child, cap, rem_budget,
                                           ref_stack):
                if len(piece) > rem_budget:
                    continue
                yield from layers(k + 1, acc + piece,
                                  rem_budget - len(piece))

        if expr.min == 0:
            yield from layers(0, "", budget)
        else:
            # min>0:只产出满足下界的串
            yield from (s for s in layers(0, "", budget)
                        if self._min_satisfied(expr, s))

    def _min_satisfied(self, expr: Repeat, s: str) -> bool:
        # 粗略校验:用匹配器确认 s 至少能按 min 次切分
        ends = self._match_repeat(expr, s, 0)
        return len(s) in ends

    def _derive_rule(self, name: str, cap: int, budget: int,
                     ref_stack: tuple[str, ...] = ()):
        yield from self._derive_expr(self.rule_expr[name], cap, budget,
                                     ref_stack)


    # ------------------------------------------------------------------ #
    # 精确前缀可延续性(递归下降,允许停在任意字符之后)
    # ------------------------------------------------------------------ #
    def can_continue(self, text: str) -> bool:
        """text 是否是某个(可能更长的)可派生串的前缀。

        核心思路:对 start 表达式求"前缀位置集合"——从位置 0 出发,
        沿 text 逐字符匹配,同时记录所有可能停在 text 内部的部分匹配。
        只要最终存在一条路径覆盖了 text 的全部字符,前缀即成立。
        每个表达式返回一个 :class:`PrefixResult`:
          * ``ends``:完整匹配该表达式后到达的位置集合;
          * ``partial``:是否存在一条路径在表达式内部(字面量/字符类
            中间)恰好把 text 消费完。
        """
        result = self._prefix_expr(self.rule_expr[self.start], text, 0, 0,
                                   ())
        return result.covers(len(text))

    def _prefix_expr(self, expr, text: str, i: int,
                     depth: int = 0,
                     call_stack: tuple[str, ...] = ()) -> "PrefixResult":
        if depth > self._depth_budget(text):
            return PrefixResult()
        if isinstance(expr, Lit):
            return self._prefix_lit(expr.text, text, i)
        if isinstance(expr, CharClass):
            return self._prefix_class(expr, text, i)
        if isinstance(expr, Ref):
            # 不消费字符的规则递归环:同一规则在调用栈上则剪枝
            if expr.name in call_stack:
                return PrefixResult()
            return self._prefix_expr(self.rule_expr[expr.name], text, i,
                                     depth + 1, call_stack + (expr.name,))
        if isinstance(expr, Seq):
            return self._prefix_seq(expr.items, text, i, depth, call_stack)
        if isinstance(expr, Alt):
            merged = PrefixResult()
            for sub in expr.items:
                merged.merge(self._prefix_expr(sub, text, i, depth,
                                               call_stack))
            return merged
        if isinstance(expr, Repeat):
            return self._prefix_repeat(expr, text, i, depth, call_stack)
        if isinstance(expr, RangeExpr):
            return self._prefix_expr(
                expand_int_range(expr.lo, expr.hi), text, i, depth,
                call_stack)
        raise AssertionError(type(expr))

    def _prefix_lit(self, lit: str, text: str, i: int) -> "PrefixResult":
        res = PrefixResult()
        if i >= len(text):
            # text 耗尽:字面量一个字符都还没消费,属合法续写(部分)
            res.partial = True
            return res
        remaining = text[i:]
        if lit.startswith(remaining):
            # text 是字面量的前缀:停在字面量中间(或正好完整)
            if len(remaining) == len(lit):
                res.ends.add(i + len(lit))
            else:
                res.partial = True
        elif remaining.startswith(lit):
            # 字面量被完整消费,后面还有 text
            res.ends.add(i + len(lit))
        return res

    def _prefix_class(self, expr: CharClass, text: str,
                      i: int) -> "PrefixResult":
        res = PrefixResult()
        if i >= len(text):
            # 字符类在等待一个字符:text 耗尽,属合法部分前缀
            res.partial = True
            return res
        if expr.match_char(text[i]):
            res.ends.add(i + 1)
        return res

    def _prefix_seq(self, items, text: str, i: int,
                    depth: int = 0,
                    call_stack: tuple[str, ...] = ()) -> "PrefixResult":
        # 逐个子表达式推进位置集合;任一阶段 partial 即整体 partial
        res = PrefixResult()
        frontier = {i}
        for sub in items:
            nxt_frontier: set[int] = set()
            for pos in frontier:
                sub_res = self._prefix_expr(sub, text, pos, depth,
                                            call_stack)
                nxt_frontier |= sub_res.ends
                if sub_res.partial:
                    res.partial = True
            frontier = nxt_frontier
            if not frontier:
                break
        res.ends |= frontier
        return res

    def _prefix_repeat(self, expr: Repeat, text: str, i: int,
                       depth: int = 0,
                       call_stack: tuple[str, ...] = ()) -> "PrefixResult":
        upper = self.cap if expr.max is None else min(expr.max, self.cap)
        upper = min(upper, max(0, self._depth_budget(text) - depth))
        res = PrefixResult()
        # layer[k]:恰好匹配 k 个 child 后的位置集合
        layers: list[set[int]] = [{i}]
        for k in range(upper):
            nxt: set[int] = set()
            for pos in layers[k]:
                child_res = self._prefix_expr(expr.child, text, pos,
                                               depth + 1, call_stack)
                nxt |= child_res.ends
                if child_res.partial:
                    # child 在内部把 text 消费完:整个 repeat 部分匹配
                    res.partial = True
            layers.append(nxt)
        for k in range(expr.min, len(layers)):
            res.ends |= layers[k]
        return res


class PrefixResult:
    __slots__ = ("ends", "partial")

    def __init__(self):
        self.ends: set[int] = set()
        self.partial: bool = False

    def merge(self, other: "PrefixResult") -> None:
        self.ends |= other.ends
        self.partial = self.partial or other.partial

    def covers(self, text_len: int) -> bool:
        return self.partial or text_len in self.ends


def brute_force_mask(interp: BruteForceInterpreter, vocab, prefix: str,
                     max_len: int) -> int:
    """暴力掩码:逐词项拼接后用递归下降前缀判定是否仍是可行前缀。

    与引擎完全独立的参考实现;直接在 AST 上做前缀匹配,
    允许停在任意字面量/符号中间,不依赖有界枚举。
    返回位图;EOS 位仅当 prefix 被完整接受时置 1。
    """
    mask = 0
    for tid, piece in enumerate(vocab.pieces):
        if tid == vocab.eos_index:
            if interp.accepts(prefix):
                mask |= vocab.bit[tid]
            continue
        candidate = prefix + piece
        if interp.can_continue(candidate):
            mask |= vocab.bit[tid]
    return mask
