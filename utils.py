import numpy as np


COL_FACTORS_NAME = ['mom_spread_6_score', 'mom_spread_12_m_1_score',
                    'mom_spread_6_score_sa', 'mom_spread_12_m_1_score_sa',
                    'leverage_z', 'leverage_z_sa', 'profit_z', 'profit_z_sa',
                    'spread_to_pd_res_score', 'spread_to_pd_res_score_sa',
                    'value_reg_richness_score', 'value_reg_richness_score_sa',
                    'momentum_score', 'carry_score', 'quality_score', 'value_score',
                    'combined_score', 'momentum_score_sa', 'carry_score_sa', 'quality_score_sa', 
                    'value_score_sa', 'combined_score_sa']

def z_score(x):
    """
    Normilises a set of values by calculating the Z-score.

    Parameters
    ----------
    x : pandas.Series, numpy.Array, list of other similar iterable object
        The value series to be normalised.
   
    Returns
    -------
    Iterable object inputted
        The value series normilised using Z-scoring.
    """
    return (x - np.mean(x)) / np.std(x)


def allocate_equally_weighted_exposure(df, factor_name, n_rows):
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    eq_w = 1 / n_rows
    df_sorted[f'{factor_name}_equally_weight'] = [eq_w] * n_rows + [0] * (df_sorted.shape[0] - n_rows)
    return df_sorted


def allocate_signal_weighted_exposure(df, factor_name, n_rows):
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    top_scores_values = df_sorted.iloc[:n_rows, :][factor_name].values
    signal_w = top_scores_values / top_scores_values.sum()
    df_sorted[f'{factor_name}_signal_weight'] = signal_w.tolist() + [0] * (df_sorted.shape[0] - n_rows)
    return df_sorted


def allocate_market_weighted_exposure(df, factor_name, n_rows):
    df_sorted = df.copy().sort_values(factor_name, ascending=False)
    top_market_values = df_sorted.iloc[:n_rows, :]['AMOUNT_OUTSTANDING'].values
    market_w = top_market_values / top_market_values.sum()
    df_sorted[f'{factor_name}_market_weight'] = market_w.tolist() + [0] * (df_sorted.shape[0] - n_rows)
    return df_sorted


def calculate_portfolio_weights(df):
    top_n_rows = df.shape[0] // 10
    for factor in COL_FACTORS_NAME:
        df = allocate_equally_weighted_exposure(df, factor, top_n_rows)
        df = allocate_signal_weighted_exposure(df, factor, top_n_rows)
        df = allocate_market_weighted_exposure(df, factor, top_n_rows)
    return df


def calculate_portfolio_returns(df):
    for factor in COL_FACTORS_NAME:
        df[f'{factor}_ret_ew'] = df[f'{factor}_equally_weight'] * df.return_excess_by_duration
        df[f'{factor}_ret_sw'] = df[f'{factor}_signal_weight'] * df.return_excess_by_duration
        df[f'{factor}_ret_mw'] = df[f'{factor}_market_weight'] * df.return_excess_by_duration
    df['benchmark_ret'] = df['benchmark_wght'] * df.return_excess_by_duration
    return df
