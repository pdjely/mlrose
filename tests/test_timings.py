""" Unit tests for timing curve for algorithms.py"""

# Author: Paul Ely
# License: BSD 3 clause
# Derived from original unit tests by GHayes

import unittest
import numpy as np
from mlrose import (OneMax, DiscreteOpt, ContinuousOpt, hill_climb,
                    random_hill_climb, simulated_annealing, genetic_alg,
                    mimic)


class TestAlgorithms(unittest.TestCase):
    """Tests for optimization algorithms."""

    @staticmethod
    def test_hill_climb_discrete_max():
        """Test hill_climb function for a discrete maximization problem"""

        problem = DiscreteOpt(5, OneMax(), maximize=True)
        _, _, curve = hill_climb(problem, 
                                 restarts=20,
                                 timing=True)

        assert (curve.shape[1] == 2)

    @staticmethod
    def test_random_hill_climb_discrete_max():
        """Test random_hill_climb function for a discrete maximization
        problem"""

        problem = DiscreteOpt(5, OneMax(), maximize=True)
        _, _, curve = random_hill_climb(problem, 
                                        max_attempts=10,
                                        restarts=20,
                                        timing=True)

        assert (curve.shape[1] == 2)

    @staticmethod
    def test_simulated_annealing_discrete_max():
        """Test simulated_annealing function for a discrete maximization
        problem"""

        problem = DiscreteOpt(5, OneMax(), maximize=True)
        _, _, curve = simulated_annealing(problem,
                                          timing=True,
                                          max_attempts=50)

        assert (curve.shape[1] == 2)

    @staticmethod
    def test_genetic_alg_discrete_max():
        """Test genetic_alg function for a discrete maximization problem"""

        problem = DiscreteOpt(5, OneMax(), maximize=True)
        _, _, curve = genetic_alg(problem, max_attempts=50, timing=True)

        assert (curve.shape[1] == 2)

    @staticmethod
    def test_mimic_discrete_max():
        """Test mimic function for a discrete maximization problem"""

        problem = DiscreteOpt(5, OneMax(), maximize=True)
        _, _, curve = mimic(problem, max_attempts=50, timing=True)

        assert (curve.shape[1] == 2)


if __name__ == '__main__':
    unittest.main()
