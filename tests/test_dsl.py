"""DSL 解析与脱糖测试。"""

import unittest

from gcdecode.dsl import (parse_grammar, desugar, ParseError, Lit, CharClass,
                          Ref, Seq, Alt, Repeat, Obj, RangeExpr,
                          expand_int_range)


class TestTokenizer(unittest.TestCase):
    def test_basic_tokens(self):
        g = parse_grammar('r ::= "ab" [a-z] r?;')
        self.assertEqual(g.start, "r")
        self.assertIn("r", g.rules)

    def test_missing_semicolon_rejected(self):
        # 规则缺少结束分号时,下一条规则名会被误当引用 —— 必须报错
        with self.assertRaises(ParseError):
            parse_grammar('a ::= "x"\nb ::= "y";')

    def test_char_class_escape_and_range(self):
        g = parse_grammar(r'r ::= [\]"] [^z];')
        expr = g.rules["r"]
        self.assertIsInstance(expr, Seq)
        cls = expr.items[0]
        self.assertIsInstance(cls, CharClass)
        self.assertTrue(cls.match_char('"'))
        neg = expr.items[1]
        self.assertTrue(neg.negated)
        self.assertTrue(neg.match_char("a"))
        self.assertFalse(neg.match_char("z"))

    def test_builtin_ws_injection(self):
        g = desugar(parse_grammar('r ::= #ws() "x";'))
        self.assertIn("__ws__", g.rules)

    def test_undefined_ref(self):
        with self.assertRaises(ParseError):
            desugar(parse_grammar("r ::= ghost;"))

    def test_duplicate_rule(self):
        with self.assertRaises(ParseError):
            parse_grammar('r ::= "x";\nr ::= "y";')


class TestIntRange(unittest.TestCase):
    def _accepted(self, expr, lo, hi):
        g = parse_grammar(f"r ::= #int({lo}, {hi});")
        return g

    def test_small_range_values(self):
        g = desugar(parse_grammar("r ::= #int(0, 3);"))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        for v in range(0, 10):
            self.assertEqual(interp.accepts(str(v)), v <= 3, v)

    def test_wider_range(self):
        g = desugar(parse_grammar("r ::= #int(7, 23);"))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        for v in range(0, 30):
            self.assertEqual(interp.accepts(str(v)), 7 <= v <= 23, v)

    def test_no_leading_zero(self):
        g = desugar(parse_grammar("r ::= #int(0, 12);"))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        self.assertFalse(interp.accepts("01"))
        self.assertFalse(interp.accepts("012"))
        self.assertTrue(interp.accepts("0"))

    def test_invalid_range(self):
        with self.assertRaises(ParseError):
            parse_grammar("r ::= #int(9, 3);")


class TestObjectSugar(unittest.TestCase):
    def test_required_and_optional_fields(self):
        text = 'r ::= {"a": "1", "b": "2" ?};'
        g = desugar(parse_grammar(text))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        self.assertTrue(interp.accepts('{"a":1}'))
        self.assertTrue(interp.accepts('{"a":1,"b":2}'))
        # 必选字段缺失必须拒绝
        self.assertFalse(interp.accepts('{"b":2}'))
        self.assertFalse(interp.accepts("{}"))
        # 字段顺序固定
        self.assertFalse(interp.accepts('{"b":2,"a":1}'))

    def test_all_required(self):
        g = desugar(parse_grammar('r ::= {"x": "1", "y": "2"};'))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        self.assertTrue(interp.accepts('{"x":1,"y":2}'))
        self.assertFalse(interp.accepts('{"x":1}'))


class TestOneOfAndRepetition(unittest.TestCase):
    def test_oneof(self):
        g = desugar(parse_grammar('r ::= #enum("RED" | "YEL");'))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        self.assertTrue(interp.accepts("RED"))
        self.assertTrue(interp.accepts("YEL"))
        self.assertFalse(interp.accepts("BLUE"))

    def test_bounded_repeat(self):
        g = desugar(parse_grammar("r ::= a{2,3};\na ::= [x];"))
        from gcdecode.nfa_interp import BruteForceInterpreter
        interp = BruteForceInterpreter(g).initialize()
        self.assertFalse(interp.accepts("x"))
        self.assertTrue(interp.accepts("xx"))
        self.assertTrue(interp.accepts("xxx"))
        self.assertFalse(interp.accepts("xxxx"))


if __name__ == "__main__":
    unittest.main()
