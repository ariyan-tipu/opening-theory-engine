"""Parse the checkable claims out of a generated explanation.

The parser is deliberately conservative: it only extracts claims it can
attribute confidently (a tight lexicon with a negation guard), because a
false positive here would punish the model for something it never said.
Anything outside the lexicon is simply not a claim and is never penalised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from opening_theory_engine.alignment.bands import (
    EvalBand,
    VerdictBand,
    parse_eval,
    parse_verdict,
)


class ClaimKind(StrEnum):
    WINS_PAWN = "wins_pawn"
    WINS_EXCHANGE = "wins_exchange"
    WINS_PIECE = "wins_piece"
    WINS_MATERIAL = "wins_material"
    LOSES_MATERIAL = "loses_material"
    GIVES_CHECK = "gives_check"
    CASTLES = "castles"
    DEVELOPS = "develops"
    CENTRE = "centre"
    FORK = "fork"
    PIN = "pin"
    BETTER_WAS = "better_was"


@dataclass(frozen=True)
class TextClaim:
    """One checkable claim found in the free-text part of an explanation."""

    kind: ClaimKind
    raw: str
    move_san: str | None = None


@dataclass(frozen=True)
class ParsedExplanation:
    """Everything the verifier needs, read out of one model response."""

    raw: str
    format_ok: bool
    verdict: VerdictBand | None
    eval_band: EvalBand | None
    line_raw: str
    line_tokens: tuple[str, ...]
    claims: tuple[TextClaim, ...]


_LABEL_RE = re.compile(
    r"^[ \t]*(verdict|position after|engine line|why)[ \t]*:[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_SAN_TOKEN_RE = re.compile(r"^(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?$")
_MOVE_NUMBER_RE = re.compile(r"\b\d+\.(?:\.\.)?|\.\.\.")

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|avoids?|concedes?|gives up|surrenders?)\b[^.;,]{0,25}$",
    re.IGNORECASE,
)

_CLAIM_PATTERNS: tuple[tuple[ClaimKind, re.Pattern[str]], ...] = (
    (
        ClaimKind.WINS_PAWN,
        re.compile(
            r"(?:wins?|winning|picks?\s+up|grabs?|nets?)\s+(?:a|the|another)?\s*pawn\b", re.I
        ),
    ),
    (
        ClaimKind.WINS_EXCHANGE,
        re.compile(r"(?:wins?|winning|picks?\s+up)\s+(?:the\s+)?exchange\b", re.I),
    ),
    (
        ClaimKind.WINS_PIECE,
        re.compile(
            r"(?:wins?|winning|picks?\s+up|nets?)\s+(?:a|the)?\s*(?:piece|knight|bishop|rook|queen)\b",
            re.I,
        ),
    ),
    (
        ClaimKind.WINS_MATERIAL,
        re.compile(r"wins?\s+material|material\s+(?:gain|advantage)|up\s+material", re.I),
    ),
    (
        ClaimKind.LOSES_MATERIAL,
        re.compile(
            r"(?:loses?|losing|drops?|hangs?|blunders?)\s+"
            r"(?:a|the|another)?\s*(?:pawn|piece|knight|bishop|rook|queen|exchange|material)\b",
            re.I,
        ),
    ),
    (
        ClaimKind.GIVES_CHECK,
        re.compile(r"(?:gives?|comes?\s+with|delivers?)\s+check\b|\bis\s+check\b", re.I),
    ),
    (ClaimKind.CASTLES, re.compile(r"\bcastl(?:es?|ing)\b", re.I)),
    (ClaimKind.DEVELOPS, re.compile(r"\bdevelop(?:s|ing|ment)?\b", re.I)),
    (
        ClaimKind.CENTRE,
        re.compile(
            r"(?:fights?\s+for|controls?|stakes?\s+a\s+claim\s+in|pressures?|"
            r"increases?\s+pressure\s+on)\s+the\s+cent(?:re|er)\b"
            r"|cent(?:re|er)\s+control\b|central\s+squares\b",
            re.I,
        ),
    ),
    (ClaimKind.FORK, re.compile(r"\bfork(?:s|ing)?\b", re.I)),
    (ClaimKind.PIN, re.compile(r"\bpin(?:s|ned|ning)?\b", re.I)),
)

_BETTER_WAS_PATTERNS = (
    re.compile(r"better\s+(?:was|is)\s+([A-Za-z0-9\-=+#]+)", re.I),
    re.compile(r"engine\s+prefers\s+([A-Za-z0-9\-=+#]+)", re.I),
)


class ClaimParser:
    """Reads model output back into structured, checkable form."""

    def parse(self, text: str) -> ParsedExplanation:
        cleaned = re.sub(r"[*`]", "", text)
        sections = self._sections(cleaned)

        verdict = parse_verdict(sections.get("verdict", ""))
        eval_band = parse_eval(sections.get("position after", ""))
        line_raw = sections.get("engine line", "").strip()
        why = sections.get("why", "").strip()

        format_ok = (
            len(sections) == 4
            and verdict is not None
            and eval_band is not None
            and bool(line_raw)
            and bool(why)
        )
        return ParsedExplanation(
            raw=text,
            format_ok=format_ok,
            verdict=verdict,
            eval_band=eval_band,
            line_raw=line_raw,
            line_tokens=self._line_tokens(line_raw),
            claims=tuple(self._text_claims(why)),
        )

    def _sections(self, text: str) -> dict[str, str]:
        matches = list(_LABEL_RE.finditer(text))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            label = match.group(1).lower()
            if label in sections:
                continue
            if label == "why":
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                sections[label] = text[match.start(2) : end].strip()
            else:
                sections[label] = match.group(2).strip()
        return sections

    @staticmethod
    def _line_tokens(line_raw: str) -> tuple[str, ...]:
        stripped = _MOVE_NUMBER_RE.sub(" ", line_raw)
        tokens = []
        for raw_token in stripped.split():
            token = raw_token.strip(".,;()")
            if token and _SAN_TOKEN_RE.match(token):
                tokens.append(token)
        return tuple(tokens)

    def _text_claims(self, why: str) -> list[TextClaim]:
        claims: list[TextClaim] = []
        seen: set[ClaimKind] = set()
        for kind, pattern in _CLAIM_PATTERNS:
            match = pattern.search(why)
            if not match or kind in seen or self._negated(why, match.start()):
                continue
            seen.add(kind)
            claims.append(TextClaim(kind=kind, raw=match.group(0)))
        for pattern in _BETTER_WAS_PATTERNS:
            match = pattern.search(why)
            if match and ClaimKind.BETTER_WAS not in seen:
                seen.add(ClaimKind.BETTER_WAS)
                claims.append(
                    TextClaim(
                        kind=ClaimKind.BETTER_WAS,
                        raw=match.group(0),
                        move_san=match.group(1).rstrip(".,;"),
                    )
                )
        return claims

    @staticmethod
    def _negated(text: str, match_start: int) -> bool:
        window = text[max(0, match_start - 30) : match_start]
        return bool(_NEGATION_RE.search(window))
