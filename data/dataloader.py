"""
Module for data loading utilities.
Author: Ryan-Rhys Griffiths
"""

import pandas as pd


def load_data(path):
    """

    :return: times, counts, uncertainties, state
    """

    df = pd.read_table(path, delimiter=' ')
    times = df['Date'].to_numpy()
    counts = df['Cts'].to_numpy()
    uncertainties = df['+-'].to_numpy()
    states = df['State'].to_numpy()

    return times, counts, uncertainties, states
