"""Phase 1: build data/processed/labels.json (class index -> Bangla character).

Source: Table 1 of Hasib et al., "BDSL 49: A comprehensive dataset of Bangla
sign language", Data in Brief 49 (2023) 109329, doi:10.1016/j.dib.2023.109329.

Dataset folders are named 0..48. File names start with the paper's label
number (1..49), and folder k always holds label k+1. audit_data.py verifies
this for every file.

PROVISIONAL: transcribed from the paper's PDF text. Please verify by eye
against results/sample_grid.png and the paper's Figure 3.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (paper label number, Bangla character, romanisation, type), in paper order.
TABLE = [
    (1, "অ", "o", "vowel"), (2, "আ", "a", "vowel"), (3, "ই", "i", "vowel"),
    (4, "উ", "u", "vowel"), (5, "এ", "e", "vowel"), (6, "ও", "o (long)", "vowel"),
    (7, "ক", "ka", "consonant"), (8, "খ", "kha", "consonant"), (9, "গ", "ga", "consonant"),
    (10, "ঘ", "gha", "consonant"), (11, "চ", "ca", "consonant"), (12, "ছ", "cha", "consonant"),
    (13, "জ", "ja", "consonant"), (14, "ঝ", "jha", "consonant"), (15, "ট", "Ta", "consonant"),
    (16, "ঠ", "Tha", "consonant"), (17, "ড", "Da", "consonant"), (18, "ঢ", "Dha", "consonant"),
    (19, "ত", "ta", "consonant"), (20, "থ", "tha", "consonant"), (21, "দ", "da", "consonant"),
    (22, "ধ", "dha", "consonant"), (23, "প", "pa", "consonant"), (24, "ফ", "pha", "consonant"),
    (25, "ব", "ba", "consonant"), (26, "ভ", "bha", "consonant"), (27, "ম", "ma", "consonant"),
    (28, "য়", "ya (antastha)", "consonant"), (29, "র", "ra", "consonant"), (30, "ল", "la", "consonant"),
    (31, "ন", "na", "consonant"), (32, "স", "sa", "consonant"), (33, "হ", "ha", "consonant"),
    (34, "ড়", "Ra", "consonant"), (35, "ং", "anusvara", "special"), (36, "ঃ", "visarga", "special"),
    (37, "০", "0", "digit"), (38, "১", "1", "digit"), (39, "২", "2", "digit"),
    (40, "৩", "3", "digit"), (41, "৪", "4", "digit"), (42, "৫", "5", "digit"),
    (43, "৬", "6", "digit"), (44, "৭", "7", "digit"), (45, "৮", "8", "digit"),
    (46, "৯", "9", "digit"), (47, "্", "hasanta", "special"), (48, "space", "space", "special"),
    (49, "ঞ", "nya", "consonant"),
]


def main():
    assert len(TABLE) == 49 and [r[0] for r in TABLE] == list(range(1, 50))
    labels = {
        "source": "Hasib et al. 2023, Data in Brief 49:109329, Table 1",
        "status": "PROVISIONAL - verify against results/sample_grid.png",
        "classes": [
            {"index": n - 1, "folder": str(n - 1), "paper_label": n,
             "bangla": ch, "roman": rom, "type": typ,
             # What the demo shows: hasanta is shown on a dotted circle so it is visible.
             "display": "◌্" if ch == "্" else ("␣" if ch == "space" else ch)}
            for n, ch, rom, typ in TABLE
        ],
    }
    out = ROOT / "data" / "processed" / "labels.json"
    out.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(labels['classes'])} classes)")


if __name__ == "__main__":
    main()
