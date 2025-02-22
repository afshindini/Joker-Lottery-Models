"""Monte Carlo analysis for lottery models."""

from dataclasses import dataclass, field
from typing import List, Tuple

import logging
import random
import pandas as pd

from .utility import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloAnalysis(Dataset):
    """Implement Monte Carlo analysis for the lottery data"""

    year: int = field(default=2025)
    week: int = field(default=1)
    day: int = field(default=1)

    def __post_init__(self) -> None:
        """Post initialization of the Monte Carlo analysis class"""
        super().__post_init__()
        self.sanity_check()
        pd.options.mode.copy_on_write = True

    def sanity_check(self) -> None:
        """Check if the period of year/week/day are consistent"""
        if (
            self.year not in [2025, 2024, 2023, 2022]
            or self.week > 52
            or self.day < 1
            or self.day > 4
        ):
            logger.error("The period is not valid. Please check year/week/day values.")

    def data_selection(self, period: str = "year") -> pd.DataFrame:
        """Select data based on the period of year/week/day"""
        if period == "year":
            grouped_data = self.data.groupby("year").get_group(self.year)
        elif period == "week":
            grouped_data = self.data.groupby("week").get_group(self.week)
        elif period == "day":
            grouped_data = self.data.groupby("day").get_group(self.day)
        else:
            grouped_data = self.data
        return grouped_data

    def monte_carlo_simulation(
        self, period: str = "year", no_simulation: int = 10000
    ) -> Tuple[List[int], List[float]]:
        """Implement Monte Carlo simulation for the lottery data"""
        digit_counts = [[0 for _ in range(10)] for _ in range(7)]
        for idx, digit in enumerate(self.headers[3:]):
            val_cnt = self.data_selection(period)[digit].value_counts()
            for i, val in enumerate(val_cnt):
                digit_counts[idx][val_cnt.index[i]] = val
        digit_probabilities = []
        for dig in digit_counts:
            total = sum(dig)
            digit_probabilities.append([val / total for val in dig])
        simulated_draws, simulated_prob = [], []
        for idx in range(7):
            suggested = random.choices(
                range(10), digit_probabilities[idx], k=no_simulation
            )  # nosec B311
            simulated_prob.append(
                suggested.count(max(set(suggested), key=suggested.count))
                / no_simulation
            )
            simulated_draws.append(max(set(suggested), key=suggested.count))
        return simulated_draws, simulated_prob
