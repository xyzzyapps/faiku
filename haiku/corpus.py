"""Classic 5-7-5 in hiragana. Mora count is the kana length (small ゃゅょっ count as one)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Poem:
    key: str
    author: str
    kana: tuple[str, str, str]
    romaji: tuple[str, str, str]
    en: str

    @property
    def lines(self) -> str:
        return " / ".join(self.kana)

    def morae(self) -> list[str]:
        return [ch for line in self.kana for ch in line]


POEMS: dict[str, Poem] = {
    "basho": Poem(
        "basho",
        "松尾芭蕉",
        ("ふるいけや", "かわずとびこむ", "みずのおと"),
        ("furu ike ya", "kawazu tobikomu", "mizu no oto"),
        "old pond — a frog jumps in, water's sound",
    ),
    "basho-cicada": Poem(
        "basho-cicada",
        "松尾芭蕉",
        ("しずかさや", "いわにしみいる", "せみのこえ"),
        ("shizukasa ya", "iwa ni shimiiru", "semi no koe"),
        "stillness — cicadas sink into the rocks",
    ),
    "buson": Poem(
        "buson",
        "与謝蕪村",
        ("なのはなや", "つきはひがしに", "ひはにしに"),
        ("nanohana ya", "tsuki wa higashi ni", "hi wa nishi ni"),
        "rape blossoms — moon east, sun west",
    ),
    "issa": Poem(
        "issa",
        "小林一茶",
        ("やせがえる", "まけるないつち", "おれとして"),
        ("yasegaeru", "makeru na icchi", "ore to shite"),
        "skinny frog, don't lose — Issa is here",
    ),
}


def get_poem(key: str = "basho") -> Poem:
    if key not in POEMS:
        raise KeyError(f"unknown poem {key!r}; try {', '.join(POEMS)}")
    return POEMS[key]
