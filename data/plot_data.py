"""
Script for plotting the Cyg-X1 lightcurve data
Author: Ryan-Rhys Griffiths 2022
"""

from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':

    df = pd.read_table('cyg_data.txt', delimiter=' ')
    times = df['Date'].to_numpy()
    counts = df['Cts'].to_numpy()
    uncertainties = df['+-'].to_numpy()
    states = df['State'].to_numpy()

    counts_state_1 = counts[states == 0]
    counts_state_2 = counts[states == 1]
    times_state_1 = times[states == 0]
    times_state_2 = times[states == 1]

    plt.scatter(times_state_1, counts_state_1, s=5, marker='+', label='State 1')
    plt.scatter(times_state_2, counts_state_2, s=5, marker='+', label='State 2')
    plt.xlabel('Time (Days)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()


