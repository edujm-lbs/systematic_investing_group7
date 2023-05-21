import pandas as pd

from utils import z_score


def calculate_momentum_signals(df):
    """
    Logic to calculate momentum signals, transform them into normalised scores,
    and combined them equally into one signal. Sub momentum factors used are:
    > 3 months spread yield difference.
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
    l_df = []
    for _isin in df.ISIN.unique():
        df_isin = df[df.ISIN == _isin].copy().sort_values('DATE')
        df_isin['mom_spread_3'] = df_isin['SPREAD_yield'].diff(periods=3)
        df_isin['mom_spread_3_score'] = df_isin.groupby(['Sector'])['mom_spread_3'].apply(lambda x: z_score(x)).fillna(0, inplace=True)
        df_isin['mom_spread_6'] = df_isin['SPREAD_yield'].diff(periods=6)
        df_isin['mom_spread_6_score'] = df_isin.groupby(['Sector'])['mom_spread_6'].apply(lambda x: z_score(x)).fillna(0, inplace=True)
        df_isin['mom_spread_12_m_1'] = df_isin['SPREAD_yield'].diff(periods=11).shift(1)
        df_isin['mom_spread_12_m_1_score'] = df_isin.groupby(['Sector'])['mom_spread_12_m_1'].apply(lambda x: z_score(x)).fillna(0, inplace=True)
        df_isin['momentum_score'] = df_isin[['mom_spread_3_score', 'mom_spread_6_score', 'mom_spread_12_m_1_score']].mean(axis=1).values
        l_df.append(df_isin)
    return pd.concat(l_df)
