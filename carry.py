from utils import z_score


def calculate_carry_signals(df):
    df['carry_score'] = z_score(df['SPREAD_yield'].to_numpy())
    return df
