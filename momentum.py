import pandas as pd

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
    l_df = []
    for _isin in df.ISIN.unique():
        df_isin = df[df.ISIN == _isin].copy().sort_values('DATE')
        df_isin['mom_spread_6'] = df_isin['SPREAD_yield'].diff(periods=6)
        df_isin['mom_spread_12_m_1'] = df_isin['SPREAD_yield'].diff(periods=11).shift(1)
        l_df.append(df_isin)
    df_out = pd.concat(l_df)
    df_out = df_out[df_out.DATE == date]
    df_out['mom_spread_6_score_sa'] = df_out.groupby(['sector'])['mom_spread_6'].apply(z_score).fillna(0)
    df_out['mom_spread_6_score'] = z_score(df_out['mom_spread_6']).fillna(0)
    df_out['mom_spread_12_m_1_score_sa'] = df_out.groupby(['sector'])['mom_spread_12_m_1'].apply(z_score).fillna(0)
    df_out['mom_spread_12_m_1_score'] = z_score(df_out['mom_spread_12_m_1']).fillna(0)
    df_out['momentum_score_sa'] = df_out[['mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa']].mean(axis=1).values
    df_out['momentum_score'] = df_out[['mom_spread_6_score', 'mom_spread_12_m_1_score']].mean(axis=1).values
    return df_out
