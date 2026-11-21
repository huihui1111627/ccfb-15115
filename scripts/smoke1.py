import sys
sys.path.insert(0, ".")

from gcdecode.dsl import parse_grammar, desugar
from gcdecode.automata import compile_grammar
from gcdecode.pda import PDA

text = """
record ::= obj;
obj    ::= "{" pair ("," pair)* "}";
pair   ::= "k" ":" [a-z]+;
"""
g = desugar(parse_grammar(text))
cg = compile_grammar(g)
pda = PDA(cg)
print("rules:", [(r.name, r.pre_min_states, r.num_states) for r in cg.rules])
print("live:", [sorted(s) for s in pda.rule_live])
for s in ['{k:abc}', '{k:ab}', '{k:a}', '{', '{k:a',
          "bad", '{k:ab},', '{k:}', '{k:ab,k:xy}']:
    print(repr(s), pda.accepts(s))
