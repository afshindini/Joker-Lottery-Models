"""Implement Markov chain analysis for lottery numbers"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any

import logging
import pandas as pd
import numpy as np

from .utility import Dataset

logger = logging.getLogger(__name__)


@dataclass
class MarkovAnalysis(Dataset):
    """Implement Markov analysis for the lottery data"""

    year: int = field(default=2025)
    week: int = field(default=1)
    day: int = field(default=1)

    def __post_init__(self) -> None:
        """Post initialization of the Markov analysis class"""
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
        """Select data based on the period of data"""
        if period == "year":
            grouped_data = self.data.groupby("year").get_group(self.year)
            logger.info("Check frequency analysis for %s %s data.", period, self.year)
        elif period == "week":
            grouped_data = self.data.groupby("week").get_group(self.week)
            logger.info("Check frequency analysis for %s %s data.", period, self.week)
        elif period == "day":
            grouped_data = self.data.groupby("day").get_group(self.day)
            logger.info("Check frequency analysis for %s %s data.", period, self.day)
        else:
            grouped_data = self.data
            logger.info("Check frequency analysis for all available data.")
        grouped_data.loc[:, "whole_number"] = grouped_data.apply(
            lambda x: str(x["d1"])
            + str(x["d2"])
            + str(x["d3"])
            + str(x["d4"])
            + str(x["d5"])
            + str(x["d6"])
            + str(x["d7"]),
            axis=1,
        )
        return grouped_data

    def _transition_matrix(self, period: str = "year") -> Any:
        """Calculate the transition matrix for the lottery data"""
        temp = self.data_selection(period)
        trans_matrix = np.zeros((10, 10))
        for row in temp.itertuples():
            for idx in range(len(row.whole_number) - 1):
                current_digit = int(row.whole_number[idx])
                next_digit = int(row.whole_number[idx + 1])
                trans_matrix[current_digit][next_digit] += 1
        return trans_matrix

    def _probability_matrix(self, period: str = "year") -> Any:
        """Convert to probabilities of transition matrix (normalize each row)"""
        matrix = self._transition_matrix(period)
        for i in range(10):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                matrix[i] /= row_sum
        return matrix

    def markov_chain(
        self, first_digit: int, period: str = "year"
    ) -> Tuple[List[int], List[float]]:
        """Calculate the Markov chain for the lottery data if you have the first digit"""
        historical_matrix = self._probability_matrix(period)
        predicted_number, probability = [first_digit], [0.1]
        for _ in range(6):
            sorted_data = np.argsort(historical_matrix[first_digit])[::-1]
            if sorted_data[0] == first_digit:
                next_digit = sorted_data[1]
            else:
                next_digit = sorted_data[0]
            predicted_number.append(int(next_digit))
            probability.append(float(historical_matrix[first_digit][next_digit]))
            first_digit = next_digit
        return predicted_number, probability
