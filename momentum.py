import pandas as pd


def calculate_momentum_signals(df):
    l_df = []
    for _isin in df.ISIN.unique():
        df_isin = df[df.ISIN == _isin].copy().sort_values('DATE')
        df_isin['mom_spread_3'] = df_isin['return_excess_by_duration'].diff(periods=3)
        df_isin['mom_spread_6'] = df_isin['return_excess_by_duration'].diff(periods=6)
        df_isin['mom_spread_12_m_1'] = df_isin['return_excess_by_duration'].diff(periods=11).shift(1)
        l_df.append(df_isin)
    return pd.concat(l_df)

# Fundamental bases momentum signals to be added
