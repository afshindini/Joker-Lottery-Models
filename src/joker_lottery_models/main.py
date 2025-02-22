"""Run the main code for Joker-Lottery-Models"""

# pylint: disable=W1202,C0209,R0914,R0801
import logging

import click
import pandas as pd

from joker_lottery_models import __version__
from joker_lottery_models.logger import config_logger
from joker_lottery_models.frequency_analysis import (
    FrequencyAnalysisPosition,
    FrequencyAnalysisGeneral,
)
from joker_lottery_models.markov_analysis import MarkovAnalysis
from joker_lottery_models.monte_carlo_analysis import MonteCarloAnalysis
from joker_lottery_models.simple_ml_predictors import RandomForestPredictor
from joker_lottery_models.complex_ml_predictors import LSTMPredictor, ARIMAPredictor

logger = logging.getLogger(__name__)


@click.command()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Shorthand for info/debug/warning/error loglevel (-v/-vv/-vvv/-vvvv)",
)
@click.option(
    "--year",
    type=int,
    default=2025,
    help="Set the year for using certain data from history",
)
@click.option(
    "--week",
    type=int,
    default=1,
    help="Set the week for using certain data from history",
)
@click.option(
    "--day", type=int, default=1, help="Set the day for using certain data from history"
)
def joker_lottery_models_cli(verbose: int, year: int, week: int, day: int) -> None:
    """Try to analyze the joker data statistically and develop AI models just for fun"""
    if verbose == 1:
        log_level = 10
    elif verbose == 2:
        log_level = 20
    elif verbose == 3:
        log_level = 30
    else:
        log_level = 40
    config_logger(log_level)

    results, guess = [], []
    first_digit = 1

    rf_pred = RandomForestPredictor("src/data/data.xlsx")
    results.append(rf_pred.predict()[0])

    arima_pred = ARIMAPredictor("src/data/data.xlsx")
    results.append(arima_pred.predict()[0])

    mnt_carlo = MonteCarloAnalysis("src/data/data.xlsx", year, week, day)
    for test in ["all", "year", "week", "day"]:
        mnt_res, _ = mnt_carlo.monte_carlo_simulation(test)
        results.append(mnt_res)

    frq_pos = FrequencyAnalysisPosition("src/data/data.xlsx", year - 1, week, day)
    for test in ["all", "year", "week", "day"]:
        if test == "day":
            first_digit = frq_pos.frequent_per_year_week_day_digits(test)[0]
        results.append(frq_pos.frequent_per_year_week_day_digits(test))

    mrk = MarkovAnalysis("src/data/data.xlsx", year, week, day)
    mrk_year_res, _ = mrk.markov_chain(first_digit, "year")
    results.append(mrk_year_res)

    frq_gen = FrequencyAnalysisGeneral("src/data/data.xlsx", year - 1, week, day)
    for test in ["all", "year", "week", "day"]:
        results.append(frq_gen.frequent_per_year_week_day(test)[0][:7])

    lstm_pred = LSTMPredictor("src/data/data.xlsx", 2025, 1, 1, 7)
    results.append(lstm_pred.predict())

    result = pd.DataFrame(results, columns=[f"d{idx}" for idx in range(1, 8)])
    for idx in range(1, 8):
        guess.append(int(result[f"d{idx}"].value_counts().idxmax()))
    click.echo(f"Final guess is: {guess}")
