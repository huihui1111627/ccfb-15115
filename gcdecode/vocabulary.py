"""词表:长短不一的字符片段 + trie 索引 + 位图掩码。

词项约定
--------
- 普通词项是任意非空字符串(字符片段),长度可以不同;
- ``eos_index`` 指向唯一的结束符词项(内部以空串标记,不进 trie);
- 位图用 Python 大整数表示,第 i 位为 1 表示词项 i 当前可用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Vocabulary", "TrieNode"]


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    # 在该节点结束的词项编号(同一节点理论上只可能有一个,保留列表稳妥)
    terminal: list[int] = field(default_factory=list)


class Vocabulary:
    def __init__(self, pieces: list[str], eos_piece: str | None = None):
        if len(pieces) != len(set(pieces)):
            raise ValueError("词表中存在重复词项")
        self.pieces: list[str] = list(pieces)
        self.eos_index: int | None = (
            self.pieces.index(eos_piece) if eos_piece is not None else None)
        self.size = len(self.pieces)
        self.root = TrieNode()
        self._build_trie()
        # 预计算每位
        self.bit = [1 << i for i in range(self.size)]
        self.all_bits = (1 << self.size) - 1

    def _build_trie(self) -> None:
        for idx, piece in enumerate(self.pieces):
            if idx == self.eos_index:
                continue  # EOS 不进 trie
            node = self.root
            for ch in piece:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = TrieNode()
                    node.children[ch] = nxt
                node = nxt
            node.terminal.append(idx)

    # ------------------------------------------------------------------ #
    # 位图工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def empty_mask() -> int:
        return 0

    def decode_mask(self, mask: int) -> list[int]:
        out: list[int] = []
        i = 0
        while mask:
            if mask & 1:
                out.append(i)
            mask >>= 1
            i += 1
        return out

    def mask_pieces(self, mask: int) -> list[str]:
        return [self.pieces[i] for i in self.decode_mask(mask)]

    def is_allowed(self, mask: int, token_id: int) -> bool:
        return bool(mask & self.bit[token_id])

    @staticmethod
    def count(mask: int) -> int:
        return mask.bit_count()
