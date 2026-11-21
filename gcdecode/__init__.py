"""gcdecode — 文法约束解码(Grammar-Constrained Decoding)组件。

模块概览
--------
- dsl:        类正则 + 递归规则的文法定义与解析
- automata:   Thompson NFA、子集法 DFA、Hopcroft 最小化
- pda:        带栈下推自动机、epsilon 闭包、共可达性分析
- vocabulary: 字符片段词表、trie、位图掩码
- scorer:     固定种子的确定性打分模型(代替真实小模型)
- decoding:   贪心 / 温度采样 / 束搜索三种约束解码
- nfa_interp: 独立的暴力参考实现(测试对照用)
- randgen:    随机文法 / 前缀生成
- report:     解码过程可视化 HTML 页面
"""

from .dsl import parse_grammar, desugar, Grammar, ParseError
from .automata import compile_grammar, CompiledGrammar
from .pda import PDA, MachineState
from .vocabulary import Vocabulary
from .scorer import SeededScorer
from .decoding import ConstrainedDecoder, DecodeTrace, DecodeStep

__all__ = [
    "parse_grammar",
    "desugar",
    "Grammar",
    "ParseError",
    "compile_grammar",
    "CompiledGrammar",
    "PDA",
    "MachineState",
    "Vocabulary",
    "SeededScorer",
    "ConstrainedDecoder",
    "DecodeTrace",
    "DecodeStep",
]
