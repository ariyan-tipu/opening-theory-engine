"""End-to-end data build: download, filter, extract, annotate, write.

This is the one function behind ``ote data build``. Everything it does is
deterministic given the config and the seed, and the slow part (engine
annotation) is cached, so re-running it is cheap.
"""

from __future__ import annotations

from opening_theory_engine.config import ProjectConfig
from opening_theory_engine.data.annotate import AnnotationCache, PositionAnnotator
from opening_theory_engine.data.dataset import DatasetBuilder, DatasetBundle
from opening_theory_engine.data.lichess import GameFilter, download_database, iter_rated_games
from opening_theory_engine.data.positions import collect_positions
from opening_theory_engine.engine.analysis import Analyser, StockfishAnalyser
from opening_theory_engine.engine.stub import StubAnalyser
from opening_theory_engine.utils import get_logger

_LOG = get_logger(__name__)


def run_data_build(
    config: ProjectConfig,
    use_stub_engine: bool = False,
    quiet: bool = False,
) -> DatasetBundle:
    """Build the full dataset for one profile and return where it landed."""
    pgn_path = download_database(config.data.pgn_url, config.paths.raw_dir, quiet=quiet)
    games = iter_rated_games(
        pgn_path,
        GameFilter.from_config(config.data),
        max_games_scanned=config.data.max_games_scanned,
        quiet=quiet,
    )
    positions = collect_positions(games, config.data, config.seed)

    analyser: Analyser
    if use_stub_engine:
        _LOG.warning("using the stub engine: fine for tests, useless for real training data")
        analyser = StubAnalyser(multipv=config.engine.multipv)
    else:
        analyser = StockfishAnalyser(config.engine)

    cache_path = config.paths.work_dir / f"annotations_{config.profile}.jsonl"
    annotator = PositionAnnotator(
        analyser,
        AnnotationCache(cache_path),
        depth=config.engine.depth,
        multipv=config.engine.multipv,
    )
    try:
        annotated = annotator.annotate_frame(positions, quiet=quiet)
    finally:
        analyser.close()

    return DatasetBuilder(config).build(annotated)
