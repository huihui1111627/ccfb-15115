"""随机文法与随机前缀生成(测试驱动)。

生成目标
--------
- 几十种结构各异的文法:嵌套对象/数组、枚举、整数范围、递归规则、
  取反字符类、可选字段、有界与无界重复;
- 每条文法配几千个前缀:一部分是合法派生的真前缀,一部分是随机噪声串;
- 随机文法默认只用有界重复(无 ``*``/``+``/``{n,}``),
  保证暴力枚举器在给定长度内覆盖整个语言,逐词项掩码才能严格比对。
"""

from __future__ import annotations

import random

from .dsl import Grammar, parse_grammar, desugar

__all__ = ["random_grammar_text", "random_grammar", "random_prefixes",
           "random_vocab_pieces"]


def random_vocab_pieces(rng: random.Random, n_pieces: int,
                        alphabet: str, eos: str = "<eos>") -> list[str]:
    """长短不一的字符片段词表,保证覆盖字母表所有单字符。"""
    # 去重并保持顺序,避免调用方传入含重复字符的 alphabet
    uniq = "".join(dict.fromkeys(alphabet))
    pieces: list[str] = [eos]
    for ch in uniq:
        if ch != " ":
            pieces.append(ch)
    seen = set(pieces)
    while len(pieces) < n_pieces:
        length = rng.choices([1, 2, 3, 4, 5], weights=[3, 4, 3, 2, 1])[0]
        s = "".join(rng.choice(uniq) for _ in range(length))
        if s not in seen and s.strip() != "":
            seen.add(s)
            pieces.append(s)
    return pieces


# --------------------------------------------------------------------------- #
# 随机文本文法
# --------------------------------------------------------------------------- #

def random_grammar_text(rng: random.Random, n_rules: int | None = None,
                        allow_recursion: bool = True) -> str:
    """生成受控的随机文本文法。

    设计上避开 DSL 不支持的组合:对象糖与括号不挂后缀(后缀只挂
    字面量/字符类/引用),内置函数统一带 # 前缀,引用只指向
    已声明规则(杜绝左递归环)。所有重复都有界,保证暴力枚举可覆盖。
    """
    n_rules = n_rules or rng.randint(3, 6)
    names = [f"r{i}" for i in range(n_rules)]
    lines: list[str] = []

    def atom(depth: int, rule_idx: int) -> str:
        choices = ["lit", "class", "ref", "oneof", "range", "obj"]
        if depth <= 0:
            choices = ["lit", "class", "oneof", "range"]
        kind = rng.choice(choices)
        if kind == "lit":
            s = "".join(rng.choice("abxy") for _ in range(rng.randint(1, 2)))
            return f'"{s}"'
        if kind == "class":
            if rng.random() < 0.2:
                return rng.choice(["[^z]", "[^ab]"])
            chars = rng.sample("abxy", k=rng.randint(1, 3))
            return "[" + "".join(chars) + "]"
        if kind == "ref":
            # 只引用前面的规则或自身(右递归安全)
            return rng.choice(names[:rule_idx + 1])
        if kind == "oneof":
            opts = rng.sample(['"RED"', '"YEL"', '"H"', '"OUT"'],
                              k=rng.randint(2, 3))
            return "#enum(" + " | ".join(opts) + ")"
        if kind == "range":
            hi = rng.randint(2, 12)
            return f"#int(0, {hi})"
        # obj:不嵌套、字段值只用简单原子,不挂后缀
        n_fields = rng.randint(1, 2)
        parts = []
        for k in range(n_fields):
            opt = " ?" if rng.random() < 0.4 else ""
            simple = rng.choice([
                '"' + "".join(rng.choice("abxy")
                              for _ in range(rng.randint(1, 2))) + '"',
                "[" + "".join(rng.sample("abxy", 2)) + "]",
                f"#int(0, {rng.randint(2, 9)})",
            ])
            parts.append(f'"f{k}": {simple}{opt}')
        return "{" + ", ".join(parts) + "}"

    def postfixed(depth: int, rule_idx: int) -> str:
        e = atom(depth, rule_idx)
        if rng.random() < 0.4:
            e += rng.choice(["?", "{0,2}", "{1,3}", "{2}"])
        return e

    def seq_expr(depth: int, rule_idx: int) -> str:
        n = rng.randint(1, 3)
        return " ".join(postfixed(depth - 1, rule_idx) for _ in range(n))

    for idx, name in enumerate(names):
        n_alt = rng.randint(1, 2)
        alts = [seq_expr(2, idx) for _ in range(n_alt)]
        lines.append(f"{name} ::= " + " | ".join(alts) + ";")
    return "\n".join(lines)


def random_grammar(rng: random.Random, **kw) -> tuple[Grammar, str]:
    for _attempt in range(50):
        text = random_grammar_text(rng, **kw)
        try:
            g = desugar(parse_grammar(text))
            return g, text
        except Exception:
            continue
    raise RuntimeError("随机文法生成连续失败")


# --------------------------------------------------------------------------- #
# 随机前缀
# --------------------------------------------------------------------------- #

def _sample_from_ast(rng: random.Random, expr, rules, budget: int,
                     cap_stack: list[int]) -> str:
    if budget <= 0:
        return ""
    from .dsl import (Lit, CharClass, Ref, Seq, Alt, Repeat,
                      RangeExpr, expand_int_range)
    if isinstance(expr, Lit):
        return expr.text if len(expr.text) <= budget else ""
    if isinstance(expr, RangeExpr):
        v = rng.randint(expr.lo, expr.hi)
        s = str(v)
        return s if len(s) <= budget else ""
    if isinstance(expr, CharClass):
        chars = []
        for lo, hi in expr.ranges:
            for cp in range(ord(lo), ord(hi) + 1):
                chars.append(chr(cp))
        if expr.negated:
            chars = [c for c in "abxy0123456789{}\",:" if expr.match_char(c)]
        return rng.choice(chars) if chars else ""
    if isinstance(expr, Ref):
        if cap_stack.count(expr.name) >= 2:
            return ""
        return _sample_from_ast(rng, rules[expr.name], rules, budget,
                                cap_stack + [expr.name])
    if isinstance(expr, Seq):
        out = ""
        for sub in expr.items:
            out += _sample_from_ast(rng, sub, rules, budget - len(out),
                                    cap_stack)
            if len(out) >= budget:
                break
        return out
    if isinstance(expr, Alt):
        return _sample_from_ast(rng, rng.choice(expr.items), rules, budget,
                                cap_stack)
    if isinstance(expr, Repeat):
        upper = expr.max if expr.max is not None else rng.randint(0, 3)
        k = rng.randint(expr.min, max(expr.min, upper))
        out = ""
        for _ in range(k):
            out += _sample_from_ast(rng, expr.child, rules,
                                    budget - len(out), cap_stack)
        return out
    raise AssertionError(type(expr))


def random_prefixes(rng: random.Random, grammar: Grammar, n: int,
                    max_len: int = 18) -> list[str]:
    """一半来自合法派生的真前缀(随机截断),一半是随机噪声。"""
    out: list[str] = []
    rules = grammar.rules
    start_expr = rules[grammar.start]
    for i in range(n):
        if i % 2 == 0:
            full = _sample_from_ast(rng, start_expr, rules, max_len, [])
            cut = rng.randint(0, len(full))
            out.append(full[:cut])
        else:
            length = rng.randint(0, max_len)
            alphabet = "abxy0123456789{}\",: "
            out.append("".join(rng.choice(alphabet) for _ in range(length)))
    return out
