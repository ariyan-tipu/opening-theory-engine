"""Alignment against engine ground truth.

The pieces here turn "the engine said X, the model said Y" into training
signal: parse the claims a generated explanation makes, check each claim
against Stockfish facts, score candidates in one vectorised JAX pass, and
keep the best-versus-worst pairs for DPO.
"""
