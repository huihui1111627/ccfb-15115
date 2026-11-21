"""文法 DSL:类正则 + 递归规则的文法定义。

文法文本示例(冰壶比赛记录)::

    record  ::= object
    object  ::= "{" #ws? "frames" #ws? ":" #ws? frame_array #ws? "}"
    ...

支持的结构
----------
- 规则:    ``name ::= expr;`` (必须以分号结束)
- 字面量:  ``"abc"`` / ``'abc'``,支持 \\n \\t \\xHH 等转义
- 字符类:  ``[a-zA-Z0-9 ]``、``[^,]``(取反,匹配任意未列出字符)
- 引用:    其他规则名,支持递归
- 序列/选择: 串联与 ``|``
- 重复:    ``e?`` ``e*`` ``e+`` ``e{n}`` ``e{n,}`` ``e{n,m}``
- 对象糖:  ``{"field": expr, "opt": expr ?}``(``?`` 标记可选字段,
           未标记的字段必须出现,字段顺序固定)
- 数组糖:  ``[expr]`` 本身即"零或多个",带界重复用后缀
- 枚举糖:  ``enum("RED" | "YEL")``
- 整数范围: ``#int(0, 8)``(展开成数字串的选择)
- 空白内置: ``#ws()``(展开为内置规则 ``__ws__`` 的引用)
- 注释:    ``#`` 后不跟标识符时为行注释

DSL 自身的空白(规则文本里)不影响生成语言;生成串里的空白必须显式写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = [
    "ParseError",
    "Expr", "Lit", "CharClass", "Ref", "Seq", "Alt", "Repeat",
    "Obj", "OneOf", "RangeExpr",
    "Grammar", "parse_grammar", "desugar",
]


class ParseError(Exception):
    """文法文本解析错误,带行列号。"""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(f"文法解析错误 (行 {line}, 列 {col}): {message}")
        self.line = line
        self.col = col


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Lit(Expr):
    text: str


@dataclass(frozen=True)
class CharClass(Expr):
    """字符类。ranges 为闭区间 (lo, hi);negated 为取反。"""
    ranges: tuple[tuple[str, str], ...]
    negated: bool = False

    def match_char(self, c: str) -> bool:
        inside = any(lo <= c <= hi for lo, hi in self.ranges)
        return (not inside) if self.negated else inside


@dataclass(frozen=True)
class Ref(Expr):
    name: str


@dataclass(frozen=True)
class Seq(Expr):
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Alt(Expr):
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Repeat(Expr):
    child: Expr
    min: int
    max: Optional[int]  # None 表示无上界


@dataclass(frozen=True)
class Obj(Expr):
    """对象糖:(键, 值表达式, 是否可选) 列表。"""
    entries: tuple[tuple[str, Expr, bool], ...]


@dataclass(frozen=True)
class OneOf(Expr):
    literals: tuple[str, ...]


@dataclass(frozen=True)
class RangeExpr(Expr):
    """#int(lo, hi) 节点,脱糖时展开。"""
    lo: int
    hi: int


# --------------------------------------------------------------------------- #
# 词法
# --------------------------------------------------------------------------- #

@dataclass
class Token:
    kind: str
    value: object
    line: int
    col: int


_SIMPLE_TOKENS = {
    "::=": "DEFINE",
    "|": "PIPE",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACK",
    "]": "RBRACK",
    "{": "LBRACE",
    "}": "RBRACE",
    ",": "COMMA",
    "?": "QUESTION",
    "*": "STAR",
    "+": "PLUS",
    ":": "COLON",
    ";": "SEMI",
}


def _unescape(text: str, line: int, col: int) -> str:
    out: list[str] = []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
              "'": "'", '"': '"', "0": "\0", "b": "\b", "f": "\f",
              "[": "[", "]": "]", "{": "{", "}": "}", "-": "-",
              "#": "#", " ": " "}
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(text):
            raise ParseError("转义符不完整", line, col + i)
        e = text[i]
        if e == "x" and i + 2 < len(text):
            out.append(chr(int(text[i + 1:i + 3], 16)))
            i += 3
        elif e == "u" and i + 4 < len(text):
            out.append(chr(int(text[i + 1:i + 5], 16)))
            i += 5
        elif e in simple:
            out.append(simple[e])
            i += 1
        else:
            raise ParseError(f"未知转义 \\{e}", line, col + i)
    return "".join(out)


def _lex_char_class(text: str, i: int, line: int, col: int) -> tuple[Token, int, int, int]:
    """读取整个 [ ... ] 并解析成字符区间集合。返回 (token, new_i, new_line, new_col)。"""
    start_line, start_col = line, col
    j = i + 1
    negated = False
    if j < len(text) and text[j] == "^":
        negated = True
        j += 1
    ranges: list[tuple[str, str]] = []
    pending: Optional[str] = None  # 待判定是否构成区间的单字符

    def read_char(pos: int) -> tuple[str, int]:
        if text[pos] == "\\":
            esc = text[pos + 1:pos + 2]
            simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
                      "]": "]", "[": "[", "-": "-", "^": "^",
                      "'": "'", '"': '"', "0": "\0", "b": "\b", "f": "\f",
                      "#": "#", " ": " "}
            if esc == "x":
                return chr(int(text[pos + 2:pos + 4], 16)), pos + 4
            if esc == "u":
                return chr(int(text[pos + 2:pos + 6], 16)), pos + 6
            if esc not in simple:
                raise ParseError(f"字符类中未知转义 \\{esc}", line, col)
            return simple[esc], pos + 2
        return text[pos], pos + 1

    while j < len(text) and text[j] != "]":
        ch, j2 = read_char(j)
        if (ch == "-" and pending is not None and j2 < len(text)
                and text[j2] != "]"):
            hi, j3 = read_char(j2)
            if pending > hi:
                raise ParseError(f"字符区间反序 {pending}-{hi}", line, col)
            ranges.append((pending, hi))
            pending = None
            j = j3
        else:
            if pending is not None:
                ranges.append((pending, pending))
            pending = ch
            j = j2
    if j >= len(text):
        raise ParseError("字符类缺少 ]", start_line, start_col)
    if pending is not None:
        ranges.append((pending, pending))
    j += 1  # 跳过 ]
    if not ranges and not negated:
        raise ParseError("空字符类", start_line, start_col)
    tok = Token("CCLS", (negated, tuple(ranges)), start_line, start_col)
    # 行/列由调用方继续推进,这里粗略计算
    consumed = text[i:j]
    new_line = line + consumed.count("\n")
    new_col = col + len(consumed) if new_line == line else (
        len(consumed) - consumed.rfind("\n") - 1)
    return tok, j, new_line, new_col


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(text)

    def advance(k: int) -> None:
        nonlocal i, line, col
        chunk = text[i:i + k]
        nl = chunk.count("\n")
        if nl:
            line += nl
            col = len(chunk) - chunk.rfind("\n")
        else:
            col += len(chunk)
        i += k

    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            advance(1)
            continue
        if ch == "#":
            # #标识符( 是内置调用,否则是注释
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            if j > i + 1 and j < n and text[j] == "(":
                name = text[i + 1:j]
                tokens.append(Token("BUILTIN", name, line, col))
                advance(j - i)
                continue
            while i < n and text[i] != "\n":
                advance(1)
            continue
        if ch in "\"'":
            quote, start_line, start_col = ch, line, col
            j = i + 1
            while j < n and text[j] != quote:
                if text[j] == "\\":
                    j += 2
                else:
                    j += 1
            if j >= n:
                raise ParseError("字符串缺少结束引号", start_line, start_col)
            raw = text[i + 1:j]
            tokens.append(Token("STRING", _unescape(raw, start_line, start_col),
                                start_line, start_col))
            advance(j + 1 - i)
            continue
        if ch == "[":
            tok, i, line, col = _lex_char_class(text, i, line, col)
            tokens.append(tok)
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(Token("NAME", text[i:j], line, col))
            advance(j - i)
            continue
        if ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            tokens.append(Token("NUMBER", int(text[i:j]), line, col))
            advance(j - i)
            continue
        if text.startswith("::=", i):
            tokens.append(Token("DEFINE", "::=", line, col))
            advance(3)
            continue
        if ch in _SIMPLE_TOKENS:
            tokens.append(Token(_SIMPLE_TOKENS[ch], ch, line, col))
            advance(1)
            continue
        raise ParseError(f"无法识别的字符 {ch!r}", line, col)
    tokens.append(Token("EOF", None, line, col))
    return tokens


# --------------------------------------------------------------------------- #
# 递归下降解析
# --------------------------------------------------------------------------- #

class _Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self, k: int = 0) -> Token:
        return self.tokens[min(self.pos + k, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def accept(self, kind: str) -> Optional[Token]:
        if self.peek().kind == kind:
            return self.next()
        return None

    def expect(self, kind: str, what: str) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise ParseError(f"期望 {what},实际得到 {tok.kind}({tok.value!r})",
                             tok.line, tok.col)
        return self.next()

    # expr = alt
    def parse_expr(self) -> Expr:
        return self.parse_alt()

    def parse_alt(self) -> Expr:
        alts = [self.parse_seq()]
        while self.accept("PIPE"):
            alts.append(self.parse_seq())
        if len(alts) == 1:
            return alts[0]
        return Alt(tuple(alts))

    def parse_seq(self) -> Expr:
        items: list[Expr] = []
        starters = {"STRING", "CCLS", "NAME", "LPAREN", "LBRACK",
                    "LBRACE", "BUILTIN"}
        while self.peek().kind in starters:
            items.append(self.parse_atom())
        if not items:
            tok = self.peek()
            raise ParseError("缺少表达式", tok.line, tok.col)
        if len(items) == 1:
            return items[0]
        return Seq(tuple(items))

    def parse_atom(self) -> Expr:
        atom = self.parse_bare_atom()
        return self.parse_postfix(atom)

    def parse_bare_atom(self) -> Expr:
        tok = self.peek()
        if tok.kind == "STRING":
            self.next()
            atom: Expr = Lit(tok.value)
        elif tok.kind == "CCLS":
            self.next()
            negated, ranges = tok.value
            atom = CharClass(ranges=ranges, negated=negated)
        elif tok.kind == "NAME":
            self.next()
            atom = Ref(tok.value)
        elif tok.kind == "LPAREN":
            self.next()
            atom = self.parse_expr()
            self.expect("RPAREN", "')'")
        elif tok.kind == "LBRACK":
            atom = self.parse_bracket_sugar()
        elif tok.kind == "LBRACE":
            atom = self.parse_object_sugar()
        elif tok.kind == "BUILTIN":
            atom = self.parse_builtin()
        else:
            raise ParseError(f"无法解析的词法记号 {tok.kind}", tok.line, tok.col)
        return atom

    def parse_bracket_sugar(self) -> Expr:
        """[expr] 是数组糖(零或多个 expr);区间用后缀 {n,m} 表达。"""
        self.expect("LBRACK", "'['")
        inner = self.parse_expr()
        self.expect("RBRACK", "']'")
        return Repeat(inner, 0, None)

    def parse_object_sugar(self) -> Expr:
        self.expect("LBRACE", "'{'")
        entries: list[tuple[str, Expr, bool]] = []
        if self.peek().kind != "RBRACE":
            while True:
                key_tok = self.expect("STRING", "字段名字符串")
                self.expect("COLON", "':'")
                # 用不带后缀的原子解析,字段级 '?' 才不会被值吞掉
                value = self.parse_bare_atom()
                optional = self.accept("QUESTION") is not None
                entries.append((key_tok.value, value, optional))
                if not self.accept("COMMA"):
                    break
        self.expect("RBRACE", "'}'")
        return Obj(tuple(entries))

    def parse_builtin(self) -> Expr:
        tok = self.next()
        name = tok.value
        self.expect("LPAREN", "'('")
        if name == "ws":
            if self.peek().kind != "RPAREN":
                raise ParseError("#ws() 不接受参数", tok.line, tok.col)
            self.expect("RPAREN", "')'")
            return Ref("__ws__")
        if name == "int":
            lo = self.expect("NUMBER", "下界整数")
            self.expect("COMMA", "','")
            hi = self.expect("NUMBER", "上界整数")
            self.expect("RPAREN", "')'")
            if lo.value < 0 or hi.value < lo.value:
                raise ParseError(f"非法整数范围 [{lo.value}, {hi.value}]",
                                 tok.line, tok.col)
            return RangeExpr(lo.value, hi.value)
        if name == "enum":
            lits: list[str] = []
            while True:
                s = self.expect("STRING", "枚举字符串字面量")
                lits.append(s.value)
                if self.accept("PIPE"):
                    continue
                break
            self.expect("RPAREN", "')'")
            if len(set(lits)) != len(lits):
                raise ParseError("枚举值重复", tok.line, tok.col)
            return OneOf(tuple(lits))
        raise ParseError(f"未知内置函数 #{name}()", tok.line, tok.col)

    def parse_postfix(self, atom: Expr) -> Expr:
        while True:
            tok = self.peek()
            if tok.kind == "QUESTION":
                self.next()
                atom = Repeat(atom, 0, 1)
            elif tok.kind == "STAR":
                self.next()
                atom = Repeat(atom, 0, None)
            elif tok.kind == "PLUS":
                self.next()
                atom = Repeat(atom, 1, None)
            elif tok.kind == "LBRACE":
                self.next()
                lo = self.expect("NUMBER", "重复下界").value
                hi: Optional[int] = lo
                if self.accept("COMMA"):
                    hi_tok = self.accept("NUMBER")
                    hi = None if hi_tok is None else hi_tok.value
                self.expect("RBRACE", "'}'")
                if hi is not None and hi < lo:
                    raise ParseError(f"重复上界 {hi} 小于下界 {lo}",
                                     tok.line, tok.col)
                atom = Repeat(atom, lo, hi)
            else:
                return atom

    def parse_grammar(self) -> Grammar:
        rules: dict[str, Expr] = {}
        order: list[str] = []
        while self.peek().kind != "EOF":
            name_tok = self.expect("NAME", "规则名")
            name = name_tok.value
            self.expect("DEFINE", "'::='")
            expr = self.parse_expr()
            # 规则必须以分号结束:否则下一条规则的规则名会被
            # 误当成上一条表达式末尾的引用
            semi = self.expect("SEMI", "';'(规则结束符)")
            if name in rules:
                raise ParseError(f"规则 {name} 重复定义", name_tok.line, name_tok.col)
            rules[name] = expr
            order.append(name)
        if not rules:
            raise ParseError("文法中没有任何规则", 1, 1)
        return Grammar(rules=rules, start=order[0], order=tuple(order))


@dataclass
class Grammar:
    rules: dict[str, Expr]
    start: str
    order: tuple[str, ...] = ()
    source: str = ""

    def rule_names(self) -> tuple[str, ...]:
        return self.order or tuple(self.rules.keys())


def parse_grammar(text: str, start: str | None = None) -> Grammar:
    parser = _Parser(text)
    g = parser.parse_grammar()
    g.source = text
    if start is not None:
        if start not in g.rules:
            raise ParseError(f"起始规则 {start} 不存在", 0, 0)
        g.start = start
    else:
        g.start = _infer_start(g)
    return g


def _infer_start(grammar: "Grammar") -> str:
    """推断起始规则:不被任何其他规则引用的规则(引用图汇点)。

    约定:文法按"被调规则在前、入口规则在后"声明,因此当存在多个
    汇点时取最后声明的那个;没有汇点(规则互相引用)时回退到最后
    一条声明的规则。
    """
    referenced: set[str] = set()
    for expr in grammar.rules.values():
        referenced.update(_walk_refs(expr))
    sinks = [n for n in grammar.rule_names() if n not in referenced]
    if sinks:
        return sinks[-1]
    return grammar.rule_names()[-1]


# --------------------------------------------------------------------------- #
# 脱糖:范围 / 枚举 / 对象糖 -> 核心结构(Lit/CharClass/Ref/Seq/Alt/Repeat)
# --------------------------------------------------------------------------- #

def _expand_equal_length(lo: int, hi: int, width: int) -> Expr:
    """展开等宽数字区间 [lo, hi] 为按位字符类的序列。"""
    slo, shi = str(lo).zfill(width), str(hi).zfill(width)
    # 递归按公共前缀切分,保证展开结果与区间完全一致
    def rec(pos: int, low: str, high: str) -> Expr:
        if pos == width:
            return Lit("")
        if low[pos] == high[pos]:
            tail = rec(pos + 1, low, high)
            seq = [Lit(low[pos]), tail]
            return Seq(tuple(seq))
        # 当前位区间 [d_lo, d_hi],后缀位分别为 9.. / 0.. / 受限
        parts: list[Expr] = []
        d_lo, d_hi = int(low[pos]), int(high[pos])
        for d in range(d_lo, d_hi + 1):
            if d == d_lo:
                suffix_lo, suffix_hi = low[pos + 1:], "9" * (width - pos - 1)
            elif d == d_hi:
                suffix_lo, suffix_hi = "0" * (width - pos - 1), high[pos + 1:]
            else:
                suffix_lo = suffix_hi = None  # type: ignore[assignment]
            head = Lit(str(d))
            if width - pos - 1 == 0:
                parts.append(head)
            elif suffix_lo is None:
                parts.append(Seq((head, CharClass((("0", "9"),)))))
                # 上面的 CharClass 只表示一位,多位需 Repeat
                parts[-1] = Seq((head, Repeat(CharClass((("0", "9"),)),
                                              width - pos - 1,
                                              width - pos - 1)))
            elif suffix_lo == suffix_hi and len(set(suffix_lo)) == 1:
                parts.append(Seq((head, Lit(suffix_lo))))
            else:
                tail_expr = _build_suffix(suffix_lo, suffix_hi)
                parts.append(Seq((head, tail_expr)))
        return Alt(tuple(parts))

    def _build_suffix(low: str, high: str) -> Expr:
        # low/high 等长,按位构造(逐字符选择/常量)
        items: list[Expr] = []
        for a, b in zip(low, high):
            if a == b:
                items.append(Lit(a))
            else:
                items.append(CharClass(((a, b),)))
        return Seq(tuple(items))

    return rec(0, slo, shi)


def expand_int_range(lo: int, hi: int) -> Expr:
    """把整数区间展开成数字串字面量/字符类的选择(无前导零,0 除外)。"""
    width_groups: dict[int, list[tuple[int, int]]] = {}
    for v in range(lo, hi + 1):
        width_groups.setdefault(len(str(v)), []).append((v, v))
    # 同宽度内合并连续区间
    alts: list[Expr] = []
    for width in sorted(width_groups):
        vals = sorted(v for v, _ in width_groups[width])
        segments: list[tuple[int, int]] = []
        seg_lo = seg_hi = vals[0]
        for v in vals[1:]:
            if v == seg_hi + 1:
                seg_hi = v
            else:
                segments.append((seg_lo, seg_hi))
                seg_lo = seg_hi = v
        segments.append((seg_lo, seg_hi))
        for a, b in segments:
            if a == b:
                alts.append(Lit(str(a)))
            else:
                alts.append(_expand_equal_length(a, b, width))
    if len(alts) == 1:
        return alts[0]
    return Alt(tuple(alts))


def _flatten(expr: Expr) -> Expr:
    if isinstance(expr, Seq):
        items: list[Expr] = []
        for e in expr.items:
            e = _flatten(e)
            if isinstance(e, Seq):
                items.extend(e.items)
            elif isinstance(e, Lit) and e.text == "":
                continue
            else:
                items.append(e)
        if not items:
            return Lit("")
        if len(items) == 1:
            return items[0]
        return Seq(tuple(items))
    if isinstance(expr, Alt):
        items = []
        for e in expr.items:
            e = _flatten(e)
            if isinstance(e, Alt):
                items.extend(e.items)
            else:
                items.append(e)
        if len(items) == 1:
            return items[0]
        return Alt(tuple(items))
    if isinstance(expr, Repeat):
        return Repeat(_flatten(expr.child), expr.min, expr.max)
    if isinstance(expr, Obj):
        return _desugar_object(expr)
    if isinstance(expr, OneOf):
        return Alt(tuple(Lit(s) for s in expr.literals))
    if isinstance(expr, RangeExpr):
        return expand_int_range(expr.lo, expr.hi)
    return expr


def _desugar_object(obj: Obj) -> Expr:
    """对象糖展开。

    必选字段恒出现;可选字段取所有子集;字段顺序按声明顺序;
    输出形如 {"a":1,"b":2}(逗号与冒号均为字面量,无自动空白)。
    """
    if len(obj.entries) > 16:
        raise ParseError("可选字段过多(>16),子集展开会爆炸", 0, 0)

    optional_idx = [i for i, (_, _, opt) in enumerate(obj.entries) if opt]
    variants: list[Expr] = []
    for mask in range(1 << len(optional_idx)):
        dropped = {optional_idx[k] for k in range(len(optional_idx))
                   if not (mask & (1 << k))}
        body_parts: list[Expr] = []
        for idx, (key, value, _opt) in enumerate(obj.entries):
            if idx in dropped:
                continue
            # 递归脱糖字段值,确保内部的 RangeExpr/嵌套对象全部展开
            value = _flatten(value)
            if body_parts:
                body_parts.append(Lit(","))
            body_parts.append(Seq((Lit('"' + key + '":'), value)))
        body: Expr = (Lit("") if not body_parts else
                      body_parts[0] if len(body_parts) == 1 else
                      Seq(tuple(body_parts)))
        variants.append(Seq((Lit("{"), body, Lit("}"))))
    if len(variants) == 1:
        return variants[0]
    return Alt(tuple(variants))


def desugar(grammar: Grammar) -> Grammar:
    """把所有语法糖脱成核心结构,并在需要时注入 __ws__ 规则。"""
    new_rules = {name: _flatten(expr) for name, expr in grammar.rules.items()}

    uses_ws = any(_contains_ref(expr, "__ws__") for expr in new_rules.values())
    if uses_ws and "__ws__" not in new_rules:
        new_rules["__ws__"] = Repeat(CharClass(((" ", " "), ("\t", "\t"),
                                                ("\n", "\n"), ("\r", "\r"))),
                                     0, None)
    order = tuple(n for n in grammar.rule_names() if n in new_rules)
    if "__ws__" not in order and "__ws__" in new_rules:
        order = order + ("__ws__",)
    # 校验引用
    for name, expr in new_rules.items():
        for ref in _walk_refs(expr):
            if ref not in new_rules:
                raise ParseError(f"规则 {name} 引用了未定义的规则 {ref}", 0, 0)
    return Grammar(rules=new_rules, start=grammar.start, order=order,
                   source=grammar.source)


def _contains_ref(expr: Expr, name: str) -> bool:
    return name in frozenset(_walk_refs(expr))


def _walk_refs(expr: Expr):
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, Ref):
            yield e.name
        elif isinstance(e, Seq):
            stack.extend(e.items)
        elif isinstance(e, Alt):
            stack.extend(e.items)
        elif isinstance(e, Repeat):
            stack.append(e.child)
