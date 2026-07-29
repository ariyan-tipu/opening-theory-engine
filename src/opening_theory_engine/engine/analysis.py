"""Stockfish wrapper: one process, typed results, explicit resource caps."""

from __future__ import annotations

import contextlib
import os
import shutil
from types import TracebackType
from typing import Protocol

import chess
import chess.engine

from opening_theory_engine.alignment.bands import MATE_SCORE_CP
from opening_theory_engine.config import EngineConfig
from opening_theory_engine.engine.facts import EngineLine, MoveFacts
from opening_theory_engine.engine.material import material_swing_cp

STOCKFISH_PATH_ENV = "STOCKFISH_PATH"


class EngineNotFoundError(RuntimeError):
    """Raised when no Stockfish binary can be located."""


class Analyser(Protocol):
    """What the rest of the project needs from an engine."""

    @property
    def name(self) -> str: ...

    def top_lines(self, board: chess.Board) -> list[EngineLine]: ...

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveFacts: ...

    def close(self) -> None: ...


def resolve_engine_path(configured: str | None = None) -> str:
    """Find a Stockfish binary: explicit config, then env var, then PATH."""
    candidates = [configured, os.environ.get(STOCKFISH_PATH_ENV)]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    raise EngineNotFoundError(
        "No Stockfish binary found. Either run 'ote engine fetch' to download the "
        f"official release, set the {STOCKFISH_PATH_ENV} environment variable, or put "
        "'stockfish' on your PATH."
    )


class StockfishAnalyser:
    """Owns a single Stockfish process. Use as a context manager.

    Thread and hash limits come from :class:`EngineConfig`, so a laptop
    profile genuinely keeps the machine quiet while a hosted profile can
    spend more.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        path = resolve_engine_path(config.path)
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._engine.configure({"Threads": config.threads, "Hash": config.hash_mb})
        self._limit = chess.engine.Limit(depth=config.depth)

    @property
    def name(self) -> str:
        return self._engine.id.get("name", "unknown engine")

    def top_lines(self, board: chess.Board) -> list[EngineLine]:
        """Analyse a position and return the top ``multipv`` lines."""
        infos = self._engine.analyse(board, self._limit, multipv=self._config.multipv)
        lines = []
        for rank, info in enumerate(infos, start=1):
            pv = tuple(info.get("pv", ()))
            if not pv:
                continue
            score = info["score"].white().score(mate_score=MATE_SCORE_CP)
            lines.append(
                EngineLine(
                    rank=rank,
                    moves_uci=tuple(move.uci() for move in pv),
                    moves_san=_san_line(board, pv),
                    cp_white=int(score),
                )
            )
        if not lines:
            raise chess.engine.EngineError("engine returned no principal variation")
        return lines

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveFacts:
        """Ground-truth assessment of one move: best line, score after, loss."""
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
        best = self.top_lines(board)[0]
        move_san = board.san(move)

        after = board.copy(stack=False)
        after.push(move)
        cp_after_white, reply_pv = self._score_after(after)

        mover_is_white = board.turn == chess.WHITE
        mover_best = best.cp_white if mover_is_white else -best.cp_white
        mover_after = cp_after_white if mover_is_white else -cp_after_white
        cp_loss = max(0, mover_best - mover_after)

        return MoveFacts(
            fen_before=board.fen(),
            move_uci=move.uci(),
            move_san=move_san,
            turn_white=mover_is_white,
            cp_best_white=best.cp_white,
            cp_after_white=cp_after_white,
            cp_loss=cp_loss,
            is_best=move.uci() == best.moves_uci[0],
            best_move_uci=best.moves_uci[0],
            best_move_san=best.moves_san[0],
            pv_uci=best.moves_uci[:8],
            pv_san=best.moves_san[:8],
            reply_pv_uci=tuple(reply.uci() for reply in reply_pv),
            reply_pv_san=_san_line(after, reply_pv),
            material_delta_cp=material_swing_cp(
                board, move, tuple(reply.uci() for reply in reply_pv)
            ),
        )

    def _score_after(self, after: chess.Board) -> tuple[int, tuple[chess.Move, ...]]:
        if after.is_checkmate():
            score = -MATE_SCORE_CP if after.turn == chess.WHITE else MATE_SCORE_CP
            return score, ()
        if after.is_game_over():
            return 0, ()
        info = self._engine.analyse(after, self._limit, multipv=1)[0]
        score = info["score"].white().score(mate_score=MATE_SCORE_CP)
        return int(score), tuple(info.get("pv", ()))[:6]

    def close(self) -> None:
        with contextlib.suppress(chess.engine.EngineError):
            self._engine.quit()

    def __enter__(self) -> StockfishAnalyser:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _san_line(
    board: chess.Board, moves: tuple[chess.Move, ...] | list[chess.Move]
) -> tuple[str, ...]:
    sans = []
    working = board.copy(stack=False)
    for move in moves:
        if move not in working.legal_moves:
            break
        sans.append(working.san(move))
        working.push(move)
    return tuple(sans)
