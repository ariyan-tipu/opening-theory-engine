"""The one prompt format used everywhere.

Training, candidate generation, evaluation and serving all build their
chat messages here, so the model always sees positions presented the same
way. If this format changes, everything retrains and re-evaluates against
the same change.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a chess opening tutor. The user gives you a position from a chess "
    "opening as a FEN string, plus the move that was played. Explain the move "
    "using exactly these four lines:\n"
    "Verdict: one of best move, good move, inaccuracy, mistake, blunder\n"
    "Position after: who stands better once the move is played\n"
    "Engine line: the expected continuation in standard algebraic notation\n"
    "Why: two or three short sentences on the ideas behind the move"
)


def build_user_message(
    opening: str,
    eco: str,
    fen_before: str,
    move_san: str,
    turn_white: bool,
) -> str:
    side = "White" if turn_white else "Black"
    return (
        f"Opening: {opening} (ECO {eco})\n"
        f"FEN before the move: {fen_before}\n"
        f"{side} played: {move_san}\n"
        "Explain this move."
    )


def build_prompt_messages(
    opening: str,
    eco: str,
    fen_before: str,
    move_san: str,
    turn_white: bool,
) -> list[dict[str, str]]:
    """System and user messages, ready for a chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(opening, eco, fen_before, move_san, turn_white),
        },
    ]
