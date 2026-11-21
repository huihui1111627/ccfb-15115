"""固定种子的确定性打分"小模型"。

不依赖任何第三方库:用可复现的伪随机数生成器(PCG 风格)生成一个
小型嵌入矩阵,每一步根据已解码前缀滚动出一个状态向量,再与词项
嵌入相乘得到 logits。同一种子 + 同一前缀永远得到同一组分数,
便于测试复现与可视化重放。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SeededScorer"]


class _Rng:
    """轻量 PCG32 伪随机数,保证跨机器可复现。"""

    def __init__(self, seed: int):
        self.state = (seed + 0x853c49e6748fea9b) & 0xFFFFFFFFFFFFFFFF
        self.inc = (0xda3e39cb94b95bdb | 1) & 0xFFFFFFFFFFFFFFFF

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) |
                (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def normal(self) -> float:
        # Box–Muller,每次消耗两个均匀数
        u1 = (self.next_u32() + 1) / 4294967297.0
        u2 = (self.next_u32() + 1) / 4294967297.0
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@dataclass
class ScorerOutput:
    logits: list[float]
    state_norm: float


class SeededScorer:
    """确定性 logits 生成器。

    参数
    ----
    vocab_size: 词项数(含 EOS)
    dim:        隐状态维度(默认 32,刻意保持"小模型")
    seed:       固定随机种子
    """

    def __init__(self, vocab_size: int, dim: int = 32, seed: int = 20261122):
        rng = _Rng(seed)
        self.vocab_size = vocab_size
        self.dim = dim
        # 词项嵌入矩阵
        self.emb = [[rng.normal() * 0.5 for _ in range(dim)]
                    for _ in range(vocab_size)]
        # 词项偏置:轻微长度偏好,制造差异化竞争
        self.bias = [rng.normal() * 0.3 for _ in range(vocab_size)]
        # 状态滚动矩阵
        self.roll = [[rng.normal() / math.sqrt(dim) for _ in range(dim)]
                     for _ in range(dim)]
        # 初始状态
        self.h0 = [rng.normal() * 0.5 for _ in range(dim)]

    def initial_state(self) -> list[float]:
        return list(self.h0)

    def step_state(self, prev: list[float], token_id: int) -> list[float]:
        """消费一个词项后滚动隐状态(确定性)。"""
        d = self.dim
        nxt = [0.0] * d
        emb = self.emb[token_id]
        for j in range(d):
            acc = 0.0
            row = self.roll[j]
            for k in range(d):
                acc += row[k] * (prev[k] + emb[k])
            # tanh 限幅,防止长序列状态爆炸
            nxt[j] = math.tanh(acc)
        return nxt

    def score(self, state: list[float]) -> ScorerOutput:
        logits: list[float] = [0.0] * self.vocab_size
        norm = 0.0
        for i in range(self.vocab_size):
            row = self.emb[i]
            acc = self.bias[i]
            for k in range(self.dim):
                acc += row[k] * state[k]
            logits[i] = acc
            norm += state[k] * state[k] if False else 0.0
        norm = math.sqrt(sum(x * x for x in state))
        return ScorerOutput(logits=logits, state_norm=norm)
