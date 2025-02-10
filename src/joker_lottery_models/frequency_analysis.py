"""Apply different frequency analysis methods to lottery data"""

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import logging
import pandas as pd

from .utility import Dataset

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAnalysis(Dataset):
    """Apply frequency analysis to the lottery data"""

    year: int = field(default=2025)
    week: int = field(default=1)
    day: int = field(default=1)

    def __post_init__(self) -> None:
        """Post initialization of the repetition analysis class"""
        super().__post_init__()
        self.sanity_check()

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
            logger.info("Check frequency analysis for %s %s data.", period, self.year)
        elif period == "week":
            grouped_data = self.data.groupby("week").get_group(self.week)
            logger.info("Check frequency analysis for  %s %s data.", period, self.week)
        elif period == "day":
            grouped_data = self.data.groupby("day").get_group(self.day)
            logger.info("Check frequency analysis for %s %s data.", period, self.day)
        else:
            grouped_data = self.data
            logger.info("Check frequency analysis for all available data.")
        return grouped_data

    def frequent_position_per_year_week_day(
        self, period: str = "year", digit: str = "d2"
    ) -> Tuple[List[int], List[float]]:
        """Find the most frequent number in each position considering data in a specific year/week/day
        and calculates their probabilities"""
        temp = self.data_selection(period)
        frequents = temp[digit].value_counts().index.tolist()
        frequents_prob = [
            val / len(temp) for val in temp[digit].value_counts().tolist()
        ]
        return frequents, frequents_prob

    def odd_even_frequency(
        self, period: str = "year", digit: str = "d1"
    ) -> Tuple[List[Literal["odd", "even"]], List[float]]:
        """Find the frequency of odd and even numbers for a specific digit"""
        temp = self.data_selection(period)
        odd_even_prob = [
            len(temp[temp[digit] % 2 == 1]) / len(temp),
            len(temp[temp[digit] % 2 == 0]) / len(temp),
        ]
        return ["odd", "even"], odd_even_prob

    def high_low_frequency(
        self, period: str = "year", digit: str = "d1"
    ) -> Tuple[List[Literal["high", "low"]], List[float]]:
        """Find the frequency of high and low numbers for a specific digit"""
        temp = self.data_selection(period)
        high_low_prob = [
            len(temp[temp[digit] > 4]) / len(temp),
            len(temp[temp[digit] <= 4]) / len(temp),
        ]
        return ["high", "low"], high_low_prob

    def frequent_digits_all(self) -> Tuple[List[int], List[float]]:
        """Find the most frequent digit in all positions nad their probabilities"""
        stacked_count = self.data[self.headers[3:]].stack().value_counts()
        probabilities = [
            val / sum(stacked_count.tolist()) for val in stacked_count.tolist()
        ]
        return stacked_count.index.tolist(), probabilities
