import pandas as pd

from utils import z_score


def calculate_momentum_signals(df):
    l_df = []
    for _isin in df.ISIN.unique():
        df_isin = df[df.ISIN == _isin].copy().sort_values('DATE')
        df_isin['mom_spread_3'] = df_isin['SPREAD_yield'].diff(periods=3)
        df['mom_spread_3_score'] = z_score(df_isin['mom_spread_3'].to_numpy())
        df_isin['mom_spread_6'] = df_isin['SPREAD_yield'].diff(periods=6)
        df['mom_spread_6_score'] = z_score(df_isin['mom_spread_6'].to_numpy())
        df_isin['mom_spread_12_m_1'] = df_isin['SPREAD_yield'].diff(periods=11).shift(1)
        df['mom_spread_12_m_1_score'] = z_score(df_isin['mom_spread_12_m_1'].to_numpy())
        l_df.append(df_isin)
    return pd.concat(l_df)

# Fundamental bases momentum signals to be added
