import numpy as np
import pandas as pd
import statsmodels.api as sm    

from utils import z_score


def calculate_momentum_signals(df, date):
    """
    Logic to calculate momentum signals, transform them into normalised scores,
    and combined them equally into one signal. Sub momentum factors used are:
    > 6 months spread yield difference.
    > 12 - 1 months spread yield difference.
    Missing values as transform to zero in normalised score.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe filtered out by a set of dates used for signal calculation.
   
    Returns
    -------
    pandas.DataFrame
        Same dataframe inputted but with extra columns for momentum signals calculated.
    """
    df['mom_spread_6'] = df.sort_values(['DATE']).groupby(['CUSIP'])['SPREAD_yield'].diff(6)
    df['mom_spread_12_m_1'] = df.sort_values(['DATE']).groupby(['CUSIP'])['SPREAD_yield'].diff(periods=11).shift(1)
    df = df[df.DATE == date]
    df['mom_spread_6_score_sa'] = df.groupby(['sector'])['mom_spread_6'].apply(z_score).fillna(0)
    df['mom_spread_6_score'] = z_score(df['mom_spread_6']).fillna(0)
    df['mom_spread_12_m_1_score_sa'] = df.groupby(['sector'])['mom_spread_12_m_1'].apply(z_score).fillna(0)
    df['mom_spread_12_m_1_score'] = z_score(df['mom_spread_12_m_1']).fillna(0)
    df['momentum_score_sa'] = df[['mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa']].mean(axis=1).values
    df['momentum_score'] = df[['mom_spread_6_score', 'mom_spread_12_m_1_score']].mean(axis=1).values
    return df
