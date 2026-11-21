"""带栈下推自动机(PDA)与共可达性分析。

运行模型
--------
- 一个"配置"为 ``(规则号, DFA 状态, 栈)``;栈帧记录返回规则、
  返回 NFA 子集(解析为 DFA 状态)和调用点状态。
- epsilon 闭包处理两类隐式转移:
  1. DFA 状态上的调用边 —— 压栈并进入被调规则起始状态;
  2. 子规则到达终态且栈非空 —— 弹栈,在父规则的返回 DFA 状态继续。
  编译期已拒绝不消费字符的调用环,闭包必然终止。
- 一个"机器状态"是闭包后配置集合的规范化表示(frozenset),
  全局编号并缓存,约束掩码按机器状态缓存。

共可达性(活性)
----------------
规则级活性用单调最小固定点求解(跨规则过程间分析):
状态 ``s`` 活,当且仅当它是终态,或存在活的字符转移,
或存在调用边且"被调规则起始态活 **且** 返回态活"。
带栈配置 ``(rule, state, stack)`` 活,当且仅当 state 活且栈中
每个返回态都活 —— 这是过程间 CFL 可达性的精确摘要刻画。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .automata import CompiledGrammar, CompiledRule

__all__ = ["PDA", "MachineState", "Frame"]


@dataclass(frozen=True)
class Frame:
    """栈帧:子规则结束后回到哪个规则的哪个 DFA 状态。"""
    return_rule: int
    return_state: int


@dataclass(frozen=True)
class MachineState:
    """规范化的机器状态:闭包后的配置集合 + 稳定编号。"""
    state_id: int
    configs: frozenset[tuple[int, int, tuple[Frame, ...]]]
    accepting: bool


class PDA:
    def __init__(self, compiled: CompiledGrammar):
        self.cg = compiled
        self.rules: list[CompiledRule] = compiled.rules
        # 机器状态驻留表
        self._intern: dict[frozenset, int] = {}
        self._states: list[MachineState] = []
        # 机器状态 -> token 位图缓存(由 decoder 注入/使用,这里只做容器)
        self.mask_cache: dict[int, int] = {}
        # 规则级活性集合
        self.rule_live: list[frozenset[int]] = []
        self._compute_liveness()
        # 起始机器状态
        self.initial = self._make_state(
            frozenset({(compiled.start_rule,
                        self.rules[compiled.start_rule].start, ())}))

    # ------------------------------------------------------------------ #
    # 共可达性(规则级最小固定点)
    # ------------------------------------------------------------------ #
    def _compute_liveness(self) -> None:
        """计算两种活性:

        - **接受活性** ``rule_accept_live[r]``:状态能否在不再消费字符的
          情况下(只走调用/返回)到达规则终态。EOS 掩码用它。
        - **前缀活性** ``rule_prefix_live[r]``:状态能否再消费至少一个
          字符(允许落入无限续写但永不结束的递归)。普通词项掩码用它 ——
          无基例尾递归(如 ``r ::= "x" r``)没有有限完整串,但其每个
          前缀都合法且可无限续写,不能被判死。
        """
        n_rules = len(self.rules)
        rev: list[dict[int, set[tuple[int, int]]]] = []
        accept_live: list[set[int]] = []
        prefix_live: list[set[int]] = []
        for r, rule in enumerate(self.rules):
            rev_r: dict[int, set[tuple[int, int]]] = {
                s: set() for s in range(rule.num_states)}
            # 接受活性种子:终态
            acc = {s for s in range(rule.num_states)
                   if rule.dfa_final[s]}
            # 前缀活性种子:有任意字符出边的状态(本层即可再消费)
            pre = {s for s in range(rule.num_states)
                   if rule.dfa_trans[s]}
            for s in range(rule.num_states):
                for _cat, t in rule.dfa_trans[s].items():
                    rev_r[t].add((r, s))
            accept_live.append(acc)
            prefix_live.append(pre)
            rev.append(rev_r)

        # 调用点:(被调规则, 调用方返回态, 调用方状态)
        call_sites: list[list[tuple[int, int, int]]] = []
        for r, rule in enumerate(self.rules):
            sites = []
            for s in range(rule.num_states):
                for callee, ret_subset in rule.dfa_calls[s]:
                    ret_state = self._resolve_return(r, ret_subset)
                    sites.append((callee, ret_state, s))
            call_sites.append(sites)
        for r, sites in enumerate(call_sites):
            for callee, ret_state, s in sites:
                # s 的活性依赖 callee.start 与 ret_state
                rev[callee][self.rules[callee].start].add((r, s))
                rev[r][ret_state].add((r, s))

        self._fixpoint(accept_live, call_sites, require_return=True)
        # 前缀活性不要求能返回(允许无限递归),只要求 callee 能消费
        self._fixpoint(prefix_live, call_sites, require_return=False)

        self.rule_accept_live = [frozenset(s) for s in accept_live]
        self.rule_prefix_live = [frozenset(s) for s in prefix_live]
        # 向后兼容旧字段名(= 接受活性)
        self.rule_live = self.rule_accept_live

    def _fixpoint(self, live: list[set[int]],
                  call_sites: list[list[tuple[int, int, int]]],
                  require_return: bool) -> None:
        """通用活性固定点。

        普通字符边:前驱随后继变活。
        调用边 (callee, ret, s):
          * require_return=True(接受活性):s 活 当且仅当
            callee.start 活 **且** ret 活;
          * require_return=False(前缀活性):s 活 当且仅当
            callee.start 活(被调方能继续消费即可)。此时 ret 的活性
            与 s 无关 —— 栈帧返回态的活性由 config_live 单独检查。
        """
        # 按语义重建反向依赖,而不是复用接受活性的反向边。
        rev: list[dict[int, set[tuple[int, int]]]] = [
            {s: set() for s in range(rule.num_states)}
            for rule in self.rules]
        for r, rule in enumerate(self.rules):
            for u in range(rule.num_states):
                for _cat, t in rule.dfa_trans[u].items():
                    rev[r][t].add((r, u))
        for r, sites in enumerate(call_sites):
            for callee, ret_state, u in sites:
                # 被调规则起始态变活 -> 调用点 u 应变活(两种语义都成立)
                rev[callee][self.rules[callee].start].add((r, u))
                if require_return:
                    # 仅接受活性:返回态活也是 u 活的必要条件
                    rev[r][ret_state].add((r, u))

        # 调用点集合,用于在激活前核对必要条件
        sites_at: list[dict[int, list[tuple[int, int]]]] = [
            {} for _ in self.rules]
        for r, sites in enumerate(call_sites):
            for callee, ret_state, u in sites:
                sites_at[r].setdefault(u, []).append((callee, ret_state))

        worklist: list[tuple[int, int]] = [
            (r, s) for r, live_r in enumerate(live) for s in live_r]
        in_work = set(worklist)
        while worklist:
            r, s = worklist.pop()
            in_work.discard((r, s))
            for (pr, ps) in rev[r][s]:
                if ps in live[pr]:
                    continue
                ok = True
                for callee, ret_state in sites_at[pr].get(ps, ()):
                    callee_live = self.rules[callee].start in live[callee]
                    if require_return:
                        if not (callee_live and ret_state in live[pr]):
                            ok = False
                            break
                    else:
                        if not callee_live:
                            ok = False
                            break
                if ok:
                    live[pr].add(ps)
                    if (pr, ps) not in in_work:
                        in_work.add((pr, ps))
                        worklist.append((pr, ps))

    def _rev_for(self, r: int, s: int, call_sites) -> set[tuple[int, int]]:
        # 已并入 _fixpoint;保留空实现以兼容潜在外部调用
        return set()

    def config_live(self, rule: int, state: int,
                    stack: tuple[Frame, ...]) -> bool:
        """前缀活性:该配置当前能否再消费至少一个字符。

        只看当前规则状态是否前缀活;栈帧返回态是否可达终态属于
        "能否结束"(接受活性/EOS)的范畴,不影响"能否继续写"。
        因此这里不检查栈帧。
        """
        return state in self.rule_prefix_live[rule]

    def config_accept_live(self, rule: int, state: int,
                           stack: tuple[Frame, ...]) -> bool:
        """接受活性:该配置能否(在某时刻)走到完整接受。"""
        if state not in self.rule_accept_live[rule]:
            return False
        for frame in stack:
            if frame.return_state not in self.rule_accept_live[frame.return_rule]:
                return False
        return True

    # ------------------------------------------------------------------ #
    # epsilon 闭包与机器状态驻留
    # ------------------------------------------------------------------ #
    def _resolve_return(self, rule_id: int,
                        nfa_subset: frozenset[int]) -> int:
        """把调用边记录的裸 NFA 返回状态集合解析为调用方 DFA 状态。

        必须先取 epsilon 闭包 —— 子集构造表的键都是闭包,
        裸子集可能恰好与另一状态的闭包键数值相同而错配。
        """
        rule = self.rules[rule_id]
        closure, _ = self._eps_closure_of(rule, nfa_subset)
        st = rule.subset_to_state.get(closure)
        if st is not None:
            return st
        raise KeyError(
            f"规则 {rule.name} 的返回闭包未在 DFA 中解析,"
            f"子集={set(nfa_subset)} 闭包={set(closure)}")

    def closure(self, configs: frozenset) -> frozenset:
        seen: set[tuple[int, int, tuple[Frame, ...]]] = set(configs)
        stack = list(configs)
        while stack:
            r, s, stk = stack.pop()
            rule = self.rules[r]
            # 1) 子规则终态 -> 弹栈返回
            if rule.dfa_final[s] and stk:
                frame = stk[-1]
                new_cfg = (frame.return_rule, frame.return_state, stk[:-1])
                if new_cfg not in seen:
                    seen.add(new_cfg)
                    stack.append(new_cfg)
            # 2) 调用边 -> 压栈进入被调规则
            for callee, ret_subset in rule.dfa_calls[s]:
                ret_state = self._resolve_return(r, ret_subset)
                frame = Frame(return_rule=r, return_state=ret_state)
                new_cfg = (callee, self.rules[callee].start, stk + (frame,))
                if new_cfg not in seen:
                    seen.add(new_cfg)
                    stack.append(new_cfg)
        return frozenset(seen)

    def _make_state(self, configs: frozenset) -> MachineState:
        closed = self.closure(configs)
        sid = self._intern.get(closed)
        if sid is not None:
            return self._states[sid]
        sid = len(self._states)
        accepting = any(r == self.cg.start_rule and self.rules[r].dfa_final[s]
                        and not stk for r, s, stk in closed)
        ms = MachineState(state_id=sid, configs=closed, accepting=accepting)
        self._intern[closed] = sid
        self._states.append(ms)
        return ms

    def state_by_id(self, state_id: int) -> MachineState:
        return self._states[state_id]

    # ------------------------------------------------------------------ #
    # 字符转移
    # ------------------------------------------------------------------ #
    def advance_char(self, ms: MachineState, ch: str) -> Optional[MachineState]:
        cat = self.cg.alphabet.char_cat.get(
            ch, self.cg.alphabet.num_cats)  # 未登记字符走通配
        nxt_configs: set = set()
        for r, s, stk in ms.configs:
            rule = self.rules[r]
            t = rule.dfa_trans[s].get(cat)
            if t is None:
                # 未登记类别只能命中通配边;子集构造时通配以显式类别外的
                # "无显式边"为条件,这里用哨兵类别复现同一语义:
                if cat == self.cg.alphabet.num_cats:
                    t = self._wild_target(rule, s, ch)
                if t is None:
                    continue
            nxt_configs.add((r, t, stk))
        if not nxt_configs:
            return None
        nxt = self._make_state(frozenset(nxt_configs))
        # 可行性检查:没有任何配置"还能续写"或"已经能接受"才判死。
        # 终态配置本身无字符出边(前缀死),但必须保留以支持 EOS。
        if not self.machine_state_viable(nxt):
            return None
        return nxt

    def machine_state_viable(self, ms: MachineState) -> bool:
        """机器状态是否可行:存在可续写配置或接受配置。"""
        for r, s, stk in ms.configs:
            if self.config_live(r, s, stk):
                return True
        # 无可续写配置时,接受配置仍可输出 EOS
        return ms.accepting

    def _wild_target(self, rule: CompiledRule, state: int,
                     ch: str) -> Optional[int]:
        """复现子集构造中通配边的语义(用于真正未登记的字符)。

        逐 NFA 状态判定:该状态在 ch 上有显式边就走显式(已在 DFA 表
        处理,不会进入这里);否则其通配边在 ch 不属于排除集时生效。
        """
        nfa_set = self._dfa_state_to_nfa_set(rule, state)
        if nfa_set is None:
            return None
        targets: set[int] = set()
        for sid in nfa_set:
            has_explicit = False
            wild_targets: set[int] = set()
            for edge in rule.nfa_states[sid].edges:
                if edge.wild:
                    if ch not in edge.wild_excluded:
                        wild_targets.add(edge.target)
                elif edge.cats is not None:
                    real_cat = self.cg.alphabet.char_cat.get(ch)
                    if real_cat is not None and real_cat in edge.cats:
                        has_explicit = True
            if not has_explicit:
                targets.update(wild_targets)
        if not targets:
            return None
        closure, _ = self._eps_closure_of(rule, frozenset(targets))
        return rule.subset_to_state.get(closure)

    def _dfa_state_to_nfa_set(self, rule: CompiledRule,
                              state: int) -> Optional[frozenset[int]]:
        for subset, dfa_state in rule.subset_to_state.items():
            if dfa_state == state:
                return subset
        return None

    def _eps_closure_of(self, rule: CompiledRule,
                        roots: frozenset[int]) -> tuple[frozenset, frozenset]:
        seen: set[int] = set(roots)
        calls: set = set()
        work = list(roots)
        while work:
            sid = work.pop()
            for edge in rule.nfa_states[sid].edges:
                if edge.cats is not None or edge.wild:
                    continue
                if edge.call_rule is not None:
                    calls.add((edge.call_rule, edge.call_return))
                    continue
                if edge.target not in seen:
                    seen.add(edge.target)
                    work.append(edge.target)
        return frozenset(seen), frozenset(calls)

    # ------------------------------------------------------------------ #
    # 整串推进(测试与报告用)
    # ------------------------------------------------------------------ #
    def consume(self, text: str,
                start: Optional[MachineState] = None) -> Optional[MachineState]:
        ms = start or self.initial
        for ch in text:
            ms = self.advance_char(ms, ch)
            if ms is None:
                return None
        return ms

    def accepts(self, text: str) -> bool:
        ms = self.consume(text)
        return ms is not None and ms.accepting

    @property
    def num_machine_states(self) -> int:
        return len(self._states)
