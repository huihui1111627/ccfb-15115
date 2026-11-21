"""自动机构造:Thompson NFA -> 子集法 DFA -> Hopcroft 最小化。

设计要点
--------
1. 每条规则先构造 Thompson NFA;字面量与普通字符类落在具体字符上,
   取反字符类 ``[^x]`` 编译为通配边(匹配任意未显式列出的字符)。
2. 规则引用编译成"调用边"(epsilon 性质,带规则号和返回 NFA 状态集合),
   不在本层展开,运行时由下推自动机 (:mod:`gcdecode.pda`) 用栈处理。
3. 字符字母表按所有规则出现的字符做全局等价类划分(类别),
   DFA 在类别上做子集构造,保证字符种类无限时状态数仍然有限。
4. DFA 经 Hopcroft 风格的划分细化最小化;调用边集合参与签名,
   不会被错误合并。
5. 每个调用点的返回位置记录为 NFA epsilon 闭包子集,并预解析到
   最小化后的 DFA 状态,供 PDA 压栈直接使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .dsl import (Grammar, Expr, Lit, CharClass, Ref, Seq, Alt, Repeat,
                  RangeExpr)
from .dsl import expand_int_range

__all__ = ["compile_grammar", "CompiledGrammar", "CompiledRule",
           "NFAEdge", "DFAState", "GrammarCompileError"]


class GrammarCompileError(Exception):
    pass


# --------------------------------------------------------------------------- #
# NFA 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class NFAEdge:
    target: int
    # None = epsilon 边;frozenset[int] = 显式字符类别集合;
    # 字符串 "WILD" = 通配(取反字符类),匹配任何没有显式边命中的类别
    cats: Optional[frozenset[int]] = None
    wild: bool = False
    # 通配边排除的字符(取反字符类 [^...])。运行时遇到未登记字符,
    # 只有不在该集合内才能走通配边。
    wild_excluded: frozenset[str] = frozenset()
    call_rule: Optional[int] = None     # 调用边:被调规则编号
    call_return: Optional[frozenset] = None  # 调用边:返回 NFA 状态集合


@dataclass
class NFAState:
    edges: list[NFAEdge] = field(default_factory=list)
    final: bool = False


@dataclass
class _Fragment:
    start: int
    end: int


# --------------------------------------------------------------------------- #
# 字母表(字符等价类)
# --------------------------------------------------------------------------- #

class Alphabet:
    """全局字符等价类划分。

    - 普通字符类/字面量里出现的字符各自归入显式类别;
    - 取反字符类不产生类别,而以通配边表达;
    - 运行时未登记字符自成一个临时类别(只可能被通配边接受)。
    """

    def __init__(self) -> None:
        self.char_cat: dict[str, int] = {}
        self.cat_chars: list[set[str]] = []

    def intern_char(self, ch: str) -> int:
        cat = self.char_cat.get(ch)
        if cat is not None:
            return cat
        cat = len(self.cat_chars)
        self.char_cat[ch] = cat
        self.cat_chars.append({ch})
        return cat

    def intern_set(self, chars: frozenset[str]) -> frozenset[int]:
        return frozenset(self.intern_char(c) for c in chars)

    def category_of(self, ch: str) -> int:
        cat = self.char_cat.get(ch)
        if cat is not None:
            return cat
        # 未登记字符:分配临时类别(编号在显式类别之后)
        cat = len(self.cat_chars) + self._extra_offset(ch)
        return cat

    def _extra_offset(self, ch: str) -> int:
        # 每个未登记字符给一个稳定但不与显式类别冲突的编号
        h = 0
        for c in ch:
            h = (h * 31 + ord(c)) & 0xFFFF
        return 1 + (h % 4096)

    @property
    def num_cats(self) -> int:
        return len(self.cat_chars)


def _class_chars(expr: CharClass) -> frozenset[str]:
    out: set[str] = set()
    for lo, hi in expr.ranges:
        cp_lo, cp_hi = ord(lo), ord(hi)
        if cp_hi - cp_lo > 2000:
            # 超大区间(如整个 Unicode 块)只登记实际可能出现的字符边界,
            # 通配语义仍由 negated/运行时类别处理;此处保护内存。
            for cp in range(cp_lo, min(cp_hi, cp_lo + 2000) + 1):
                out.add(chr(cp))
        else:
            for cp in range(cp_lo, cp_hi + 1):
                out.add(chr(cp))
    return frozenset(out)


# --------------------------------------------------------------------------- #
# Thompson 构造
# --------------------------------------------------------------------------- #

class _NFABuilder:
    def __init__(self, alphabet: Alphabet, rule_index: dict[str, int]):
        self.states: list[NFAState] = []
        self.alpha = alphabet
        self.rule_index = rule_index

    def new_state(self, final: bool = False) -> int:
        idx = len(self.states)
        self.states.append(NFAState(final=final))
        return idx

    def add_eps(self, src: int, dst: int) -> None:
        self.states[src].edges.append(NFAEdge(target=dst))

    def build(self, expr: Expr) -> _Fragment:
        if isinstance(expr, Lit):
            start = self.new_state()
            prev = start
            if expr.text == "":
                end = self.new_state()
                self.add_eps(prev, end)
                return _Fragment(start, end)
            for ch in expr.text:
                cat = self.alpha.intern_char(ch)
                nxt = self.new_state()
                self.states[prev].edges.append(
                    NFAEdge(target=nxt, cats=frozenset({cat})))
                prev = nxt
            return _Fragment(start, prev)

        if isinstance(expr, CharClass):
            start = self.new_state()
            end = self.new_state()
            if expr.negated:
                # 登记被排除字符,使它们获得显式类别;通配边记录排除集,
                # 运行时遇到未登记字符时据此拒绝被排除的字符。
                excluded = _class_chars(
                    CharClass(ranges=expr.ranges, negated=False))
                for ch in excluded:
                    self.alpha.intern_char(ch)
                self.states[start].edges.append(
                    NFAEdge(target=end, wild=True,
                            wild_excluded=excluded))
            else:
                cats = self.alpha.intern_set(_class_chars(expr))
                self.states[start].edges.append(
                    NFAEdge(target=end, cats=cats))
            return _Fragment(start, end)

        if isinstance(expr, Ref):
            rule_id = self.rule_index[expr.name]
            start = self.new_state()
            ret = self.new_state()
            # 返回点就是调用边的目标状态本身(裸子集);
            # PDA 解析时取 epsilon 闭包映射到 DFA 状态。
            # 不能在此处预先闭包,否则会把调用之后的文法吞进返回点。
            self.states[start].edges.append(
                NFAEdge(target=ret, call_rule=rule_id,
                        call_return=frozenset({ret})))
            return _Fragment(start, ret)

        if isinstance(expr, Seq):
            frags = [self.build(e) for e in expr.items]
            for a, b in zip(frags, frags[1:]):
                self.add_eps(a.end, b.start)
            return _Fragment(frags[0].start, frags[-1].end)

        if isinstance(expr, Alt):
            start = self.new_state()
            end = self.new_state()
            for sub in expr.items:
                frag = self.build(sub)
                self.add_eps(start, frag.start)
                self.add_eps(frag.end, end)
            return _Fragment(start, end)

        if isinstance(expr, Repeat):
            return self._build_repeat(expr)

        if isinstance(expr, RangeExpr):
            # 双保险:正常情况下脱糖阶段已展开
            return self.build(expand_int_range(expr.lo, expr.hi))

        raise GrammarCompileError(f"无法编译的 AST 节点: {type(expr).__name__}")

    def _build_repeat(self, expr: Repeat) -> _Fragment:
        # 有界部分展开为 min 个副本
        start = self.new_state()
        prev_end = start
        copies: list[_Fragment] = []
        for _ in range(expr.min):
            frag = self.build(expr.child)
            self.add_eps(prev_end, frag.start)
            prev_end = frag.end
            copies.append(frag)
        if expr.max is None:
            # (child)* 循环:从当前尾连一个 child 副本并回边
            loop_start = self.new_state()
            self.add_eps(prev_end, loop_start)
            frag = self.build(expr.child)
            self.add_eps(loop_start, frag.start)
            end = self.new_state()
            self.add_eps(frag.end, loop_start)  # 回边,支持任意多次
            self.add_eps(loop_start, end)       # 退出循环
            return _Fragment(start, end)
        extra = expr.max - expr.min
        if extra == 0:
            return _Fragment(start, prev_end)
        # 可选副本 0..extra 个
        end = self.new_state()
        self.add_eps(prev_end, end)
        cur = prev_end
        for _ in range(extra):
            frag = self.build(expr.child)
            self.add_eps(cur, frag.start)
            self.add_eps(frag.end, end)
            cur = frag.end
        return _Fragment(start, end)

    # --- epsilon 闭包(含调用边记录) --- #
    def eps_closure(self, roots: frozenset[int]):
        """返回 (闭包状态集, 闭包内发现的调用边去重集合)。

        调用边不在 NFA 层跨越(跨规则由 PDA 处理),只记录其目标规则与
        返回状态集合。
        """
        seen: set[int] = set(roots)
        calls: set[tuple[int, frozenset[int]]] = set()
        stack = list(roots)
        while stack:
            sid = stack.pop()
            for edge in self.states[sid].edges:
                if edge.cats is not None or edge.wild:
                    continue
                if edge.call_rule is not None:
                    calls.add((edge.call_rule, edge.call_return))
                    continue
                if edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return frozenset(seen), frozenset(calls)


# --------------------------------------------------------------------------- #
# DFA 状态(最小化前)
# --------------------------------------------------------------------------- #

@dataclass
class DFAState:
    nfa_set: frozenset[int]
    final: bool
    # cat -> 下一 DFA 状态 id(构造期);通配兜底在 move 时处理
    trans: dict[int, int] = field(default_factory=dict)
    # 该状态 epsilon 闭包里出现的调用边:(规则号, 返回 NFA 子集)
    calls: frozenset[tuple[int, frozenset[int]]] = frozenset()


# --------------------------------------------------------------------------- #
# 编译后的规则(最小化 DFA)
# --------------------------------------------------------------------------- #

@dataclass
class CompiledRule:
    name: str
    nfa_states: list[NFAState]
    # 最小化 DFA
    dfa_trans: list[dict[int, int]]          # state -> cat -> state
    dfa_final: list[bool]
    dfa_calls: list[frozenset[tuple[int, frozenset[int]]]]
    start: int
    # NFA epsilon 闭包子集 -> 最小化 DFA 状态(返回点解析)
    subset_to_state: dict[frozenset[int], int]
    num_states: int
    # 最小化前的状态数(用于报告与测试)
    pre_min_states: int
    # 本规则可接受的全部字符类别(显式 + 通配)
    all_cats: frozenset[int]


@dataclass
class CompiledGrammar:
    rules: list[CompiledRule]
    rule_by_name: dict[str, int]
    alphabet: Alphabet
    start_rule: int
    grammar: Grammar
    # epsilon 调用图(编译期终止性检查用)
    eps_call_graph: dict[int, frozenset[int]]


def _build_dfa(builder: _NFABuilder, num_cats: int,
               extra_roots: frozenset[frozenset[int]] = frozenset()
               ) -> tuple[list[DFAState], dict[frozenset[int], int]]:
    """子集构造。新状态可能在任意一轮扫描中出现,用固定点扫描到稳定。

    ``extra_roots`` 是其他规则调用本规则时记录的返回闭包,
    必须全部建成 DFA 状态,否则 PDA 压栈后无法解析返回点。
    """
    start_closure, start_calls = builder.eps_closure(frozenset({0}))
    states: list[DFAState] = []
    index: dict[frozenset[int], int] = {}

    def register(closure: frozenset[int],
                 calls: frozenset[tuple[int, frozenset[int]]]) -> None:
        if closure not in index:
            index[closure] = len(states)
            states.append(DFAState(
                nfa_set=closure,
                final=any(builder.states[s].final for s in closure),
                calls=calls))

    register(start_closure, start_calls)
    # 外部调用点的返回闭包(已在本规则的 NFA 状态空间内)
    for root in extra_roots:
        closure, calls = builder.eps_closure(root)
        register(closure, calls)

    # worklist:每个 DFA 状态只展开一次;新状态入队等待展开
    worklist = list(range(len(states)))
    while worklist:
        dfa = states[worklist.pop()]
        # 逐 NFA 状态、逐类别收集目标。通配边(取反字符类)的语义是
        # "该 NFA 状态在这个字符上没有显式边时才生效",因此必须按
        # 源 NFA 状态分别判定,不能用全局 else —— 否则两个并行 NFA
        # 状态里一个走显式边、一个走通配边的歧义路径会被错误丢弃。
        per_cat: dict[int, set[int]] = {}
        for sid in dfa.nfa_set:
            explicit_cats: set[int] = set()
            wild_specs: list[tuple[int, frozenset[str]]] = []
            for edge in builder.states[sid].edges:
                if edge.cats is not None:
                    for cat in edge.cats:
                        explicit_cats.add(cat)
                        per_cat.setdefault(cat, set()).add(edge.target)
                elif edge.wild:
                    wild_specs.append((edge.target, edge.wild_excluded))
            if wild_specs:
                # 被排除字符对应的类别集合
                excluded_cats = set()
                for _tgt, excluded in wild_specs:
                    for ch in excluded:
                        cat = builder.alpha.char_cat.get(ch)
                        if cat is not None:
                            excluded_cats.add(cat)
                for cat in range(num_cats):
                    if cat in explicit_cats or cat in excluded_cats:
                        continue
                    for tgt, _ex in wild_specs:
                        per_cat.setdefault(cat, set()).add(tgt)
        trans: dict[int, int] = {}
        for cat, targets in per_cat.items():
            if not targets:
                continue
            closure, calls = builder.eps_closure(frozenset(targets))
            if closure not in index:
                register(closure, calls)
                worklist.append(index[closure])
            trans[cat] = index[closure]
        dfa.trans = trans
    return states, index


def _minimize(states: list[DFAState], num_cats: int) -> tuple[
        list[int], int]:
    """Hopcroft 风格划分细化(标准固定点实现)。

    初始按 (final, 调用边签名) 分块,随后反复用"对每个字符类别转移到
    哪个块"的签名重分,直到划分稳定。关键是每轮基于**上一轮**的块
    编号计算签名、整轮原子替换,避免块在一轮内部被提前合并而丢失区分。
    返回 (旧状态->新状态映射, 新状态数)。
    """
    n = len(states)

    def block_key(s: DFAState):
        return (s.final, s.calls)

    # 初始划分
    part_of: list[int] = [0] * n
    init_groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(states):
        init_groups.setdefault(block_key(s), []).append(i)
    for pid, members in enumerate(init_groups.values()):
        for i in members:
            part_of[i] = pid

    while True:
        # 用上一轮的 part_of 计算每个状态的完整签名
        signatures: dict[tuple, list[int]] = {}
        for i, s in enumerate(states):
            trans_sig = tuple(
                (cat, part_of[t])
                for cat, t in sorted(s.trans.items()))
            sig = (s.final, s.calls, trans_sig)
            signatures.setdefault(sig, []).append(i)
        new_part = [0] * n
        for pid, members in enumerate(signatures.values()):
            for i in members:
                new_part[i] = pid
        if new_part == part_of:
            part_of = new_part
            break
        part_of = new_part

    return part_of, max(part_of) + 1


def _check_termination(compiled_rules: list[CompiledRule],
                       rule_by_name: dict[str, int]) -> dict[int, frozenset[int]]:
    """检查 epsilon 驱动的调用环(会导致闭包无限压栈)。

    关键:只有"无法绕过"的调用边才危险。可选结构(``e?``/``e*``)里的
    调用存在一条 epsilon 旁路,不会强制无限压栈。因此这里从每条规则的
    起始状态做 epsilon 遍历,记录沿途遇到的调用边;只有当某条调用边
    在规则的每条 epsilon 路径上都出现(必选调用)时,才纳入必选调用图,
    图上有环才判定为不终止的左递归。
    """
    graph: dict[int, set[int]] = {i: set() for i in range(len(compiled_rules))}
    for rid, rule in enumerate(compiled_rules):
        # 从状态 0 出发,epsilon 可达的全部调用边
        reachable_calls: set[int] = set()
        seen: set[int] = set()
        stack = [0]
        while stack:
            sid = stack.pop()
            if sid in seen:
                continue
            seen.add(sid)
            for edge in rule.nfa_states[sid].edges:
                if edge.cats is not None or edge.wild:
                    continue
                if edge.call_rule is not None:
                    reachable_calls.add(edge.call_rule)
                    continue
                stack.append(edge.target)
        # 判定每条可达调用是否"必选":从起点能否不经过该调用边而到达终态。
        # 能绕过 -> 可选,不进必选图;不能绕过 -> 进图参与环检测。
        final_states = {i for i, st in enumerate(rule.nfa_states)
                        if st.final}
        for callee in reachable_calls:
            if not _can_reach_final_without(rule, 0, final_states, callee):
                graph[rid].add(callee)

    color: dict[int, int] = {i: 0 for i in graph}

    def dfs(u: int, path: list[int]) -> None:
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1:
                cycle = path[path.index(v):] + [v] if v in path else [v]
                names = [compiled_rules[k].name for k in cycle]
                raise GrammarCompileError(
                    "文法存在不消费字符的必选递归调用环(左递归),"
                    "无法保证终止: " + " -> ".join(names))
            if color[v] == 0:
                dfs(v, path + [v])
        color[u] = 2

    for rid in graph:
        if color[rid] == 0:
            dfs(rid, [rid])
    return {k: frozenset(v) for k, v in graph.items()}


def _can_reach_final_without(rule: CompiledRule, start: int,
                             final_states: set[int],
                             avoid_callee: int) -> bool:
    """是否存在一条只走 epsilon 边、且不调用 avoid_callee 的路径到达终态。"""
    seen: set[int] = set()
    stack = [start]
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        seen.add(sid)
        if sid in final_states:
            return True
        for edge in rule.nfa_states[sid].edges:
            if edge.cats is not None or edge.wild:
                continue
            if edge.call_rule is not None:
                if edge.call_rule == avoid_callee:
                    continue  # 绕过这条调用
                continue  # 其他调用也不跨越(过程内分析)
            stack.append(edge.target)
    return False


def compile_grammar(grammar: Grammar) -> CompiledGrammar:
    """把脱糖后的文法编译为最小化 DFA + 调用元数据。"""
    rule_by_name = {name: i for i, name in enumerate(grammar.rule_names())}
    alphabet = Alphabet()

    # 第一遍:构造各规则 NFA(字母表在构造中收集)
    builders: list[_NFABuilder] = []
    for name in grammar.rule_names():
        b = _NFABuilder(alphabet, rule_by_name)
        frag = b.build(grammar.rules[name])
        b.states[frag.end].final = True
        builders.append(b)

    num_cats = alphabet.num_cats

    # 第一遍半:收集每条规则作为"调用方"时,其调用边返回点的 NFA 闭包。
    # 返回点属于调用方自己的 NFA 状态空间,必须在调用方规则的 DFA 里
    # 建成状态,PDA 弹栈时才能解析回调用方的 DFA 状态。
    extra_roots: list[set[frozenset[int]]] = [
        set() for _ in builders]
    for caller_id, b in enumerate(builders):
        for st in b.states:
            for edge in st.edges:
                if edge.call_rule is not None:
                    extra_roots[caller_id].add(edge.call_return)

    # 第二遍:每条规则做子集 DFA + 最小化
    compiled: list[CompiledRule] = []
    for rid, (name, b) in enumerate(zip(grammar.rule_names(), builders)):
        raw_states, subset_index = _build_dfa(
            b, num_cats, frozenset(extra_roots[rid]))
        pre_min = len(raw_states)
        part_id, new_n = _minimize(raw_states, num_cats)

        trans: list[dict[int, int]] = [{} for _ in range(new_n)]
        finals = [False] * new_n
        calls: list[frozenset] = [frozenset() for _ in range(new_n)]
        for old_id, state in enumerate(raw_states):
            new_id = part_id[old_id]
            finals[new_id] = state.final
            calls[new_id] = state.calls
            for cat, t in state.trans.items():
                trans[new_id][cat] = part_id[t]
        # 子集 -> 最小化状态
        subset_to_state = {subset: part_id[old]
                           for subset, old in subset_index.items()}
        start_state = part_id[subset_index[next(iter(subset_index))]]
        all_cats = frozenset(range(num_cats))
        compiled.append(CompiledRule(
            name=name, nfa_states=b.states, dfa_trans=trans,
            dfa_final=finals, dfa_calls=calls, start=start_state,
            subset_to_state=subset_to_state, num_states=new_n,
            pre_min_states=pre_min, all_cats=all_cats))

    eps_graph = _check_termination(compiled, rule_by_name)
    return CompiledGrammar(
        rules=compiled, rule_by_name=rule_by_name, alphabet=alphabet,
        start_rule=rule_by_name[grammar.start], grammar=grammar,
        eps_call_graph=eps_graph)
