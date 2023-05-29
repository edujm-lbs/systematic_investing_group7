import warnings

import numpy as np
import pandas as pd

from carry import calculate_carry_signals
from momentum import calculate_momentum_signals
from quality import quality_calc
from utils import calculate_portfolio_weights, calculate_portfolio_returns
from value import calculate_value_signals

########################################################################
# Script containing the main logic for the investment strategy process #
# The script outputs two csv files with the set of securities selected #
########################################################################

warnings.filterwarnings(action='once')

# Include any relevant field used for your analysis - avoid having such a large DataFrame
REL_COL = ['DATE', 'CUSIP', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'SPREAD_yield',
           'return_excess_by_duration', 'datadate', 'HSICCD', 'dlc', 'dltt',
           'mib', 'upstk', 'che', 'mv', 'dt', 'gp', 'at', 'ret_var_movstd_yrl',
           'mkvalt', 'DURATION', 'TMT', 'N_SP', 'lead_EXCESS_RET', 'lead_TOT_RET']

DIR = 'bondret_data17e.sas7bdat'

df_data = pd.read_sas(DIR).loc[:,REL_COL]
df_data = df_data[df_data['datadate'].notna()]

df_data['sector'] = df_data['HSICCD'].astype(str).str[0]
df_data['dts'] = df_data['DURATION'] * df_data['SPREAD_yield']
df_data['TMT_2'] = df_data['TMT'] ** 2

l_df = []
for date in df_data.DATE.sort_values().unique()[13:]:
    print(date)
    date_prev = date - np.timedelta64(500, 'D')
    # For momentum signal calculation we require trailing dates.
    df_dt = df_data[(df_data.DATE > date_prev) & (df_data.DATE <= date)].copy()
    # 1. Calculation of signals both sector adjusted ("sa") and raw. All signals are DTS adjusted.
    ## > Momentum, including 6-months and 12-1-months momentum signals. 
    df_dt_m = calculate_momentum_signals(df_dt, date)
    ## > Carry, including only one signal.
    df_dt_m_c = calculate_carry_signals(df_dt_m)
    ## > Quality, including profitability and leverage signals.
    df_dt_m_c_q = quality_calc(df_dt_m_c)
    ## > Value, including predicted spread differential and spread to PD signals.
    df_dt_m_c_q_v = calculate_value_signals(df_dt_m_c_q)
    ## 2. Combine all Style combined Z-Scores on to one
    df_dt_m_c_q_v['combined_score'] = \
        df_dt_m_c_q_v[['momentum_score', 'carry_score', 'quality_score', 'value_score']].mean(axis=1)
    df_dt_m_c_q_v['combined_score_sa'] = \
        df_dt_m_c_q_v[['momentum_score_sa', 'carry_score_sa', 'quality_score_sa', 'value_score_sa']].mean(axis=1)
    # 3. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_q_v_w = calculate_portfolio_weights(df_dt_m_c_q_v)
    # 4. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_q_v_w['benchmark_wght'] = \
        df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'] / df_dt_m_c_q_v_w['AMOUNT_OUTSTANDING'].sum()
    l_df.append(df_dt_m_c_q_v_w)

df_final = pd.concat(l_df)

df_final = calculate_portfolio_returns(df_final)
df_final.to_csv('strategy_portfolio_output.csv')

COL_RET = ['mom_spread_6_score_ret_ew_le', 'mom_spread_6_score_ret_sw_le', 'mom_spread_6_score_ret_mw_le',
           'mom_spread_6_score_ret_ew_ltr', 'mom_spread_6_score_ret_sw_ltr', 'mom_spread_6_score_ret_mw_ltr',
           'mom_spread_6_score_sa_ret_ew_le', 'mom_spread_6_score_sa_ret_sw_le', 'mom_spread_6_score_sa_ret_mw_le',
           'mom_spread_6_score_sa_ret_ew_ltr', 'mom_spread_6_score_sa_ret_sw_ltr', 'mom_spread_6_score_sa_ret_mw_ltr',
           'mom_spread_12_m_1_score_ret_ew_le', 'mom_spread_12_m_1_score_ret_sw_le',
           'mom_spread_12_m_1_score_ret_ew_ltr', 'mom_spread_12_m_1_score_ret_sw_ltr',
           'mom_spread_12_m_1_score_sa_ret_ew_le', 'mom_spread_12_m_1_score_sa_ret_sw_le',
           'mom_spread_12_m_1_score_sa_ret_ew_ltr', 'mom_spread_12_m_1_score_sa_ret_sw_ltr',
           'mom_spread_12_m_1_score_ret_mw_le', 'momentum_score_ret_ew_le', 'momentum_score_ret_sw_le',
           'mom_spread_12_m_1_score_ret_mw_ltr', 'momentum_score_ret_ew_ltr', 'momentum_score_ret_sw_ltr',
           'mom_spread_12_m_1_score_sa_ret_mw_le', 'momentum_score_sa_ret_ew_le', 'momentum_score_sa_ret_sw_le',
           'mom_spread_12_m_1_score_sa_ret_mw_ltr', 'momentum_score_sa_ret_ew_ltr', 'momentum_score_sa_ret_sw_ltr',
           'momentum_score_ret_mw_le', 'carry_score_ret_ew_le', 'carry_score_ret_sw_le', 'carry_score_ret_mw_le',
           'momentum_score_ret_mw_ltr', 'carry_score_ret_ew_ltr', 'carry_score_ret_sw_ltr', 'carry_score_ret_mw_ltr',
           'momentum_score_sa_ret_mw_le', 'carry_score_sa_ret_ew_le', 'carry_score_sa_ret_sw_le', 'carry_score_sa_ret_mw_le',
           'momentum_score_sa_ret_mw_ltr', 'carry_score_sa_ret_ew_ltr', 'carry_score_sa_ret_sw_ltr', 'carry_score_sa_ret_mw_ltr',
           'leverage_z_ret_ew_le',	'leverage_z_ret_sw_le', 'leverage_z_ret_mw_le',
           'leverage_z_ret_ew_ltr',	'leverage_z_ret_sw_ltr', 'leverage_z_ret_mw_ltr',
           'leverage_z_sa_ret_ew_le', 'leverage_z_sa_ret_sw_le', 'leverage_z_sa_ret_mw_le',
           'leverage_z_sa_ret_ew_ltr', 'leverage_z_sa_ret_sw_ltr', 'leverage_z_sa_ret_mw_ltr',
           'profit_z_ret_ew_le', 'profit_z_ret_sw_le', 'profit_z_ret_mw_le',
           'profit_z_ret_ew_ltr', 'profit_z_ret_sw_ltr', 'profit_z_ret_mw_ltr',
           'profit_z_sa_ret_ew_le', 'profit_z_sa_ret_sw_le', 'profit_z_sa_ret_mw_le',
           'profit_z_sa_ret_ew_ltr', 'profit_z_sa_ret_sw_ltr', 'profit_z_sa_ret_mw_ltr',
           'quality_score_ret_ew_le', 'quality_score_ret_sw_le', 'quality_score_ret_mw_le',
           'quality_score_sa_ret_ew_ltr', 'quality_score_sa_ret_sw_ltr', 'quality_score_sa_ret_mw_ltr',
           'spread_to_pd_res_score_ret_ew_le', 'spread_to_pd_res_score_ret_sw_le', 'spread_to_pd_res_score_ret_mw_le',
           'spread_to_pd_res_score_ret_ew_ltr', 'spread_to_pd_res_score_ret_sw_ltr', 'spread_to_pd_res_score_ret_mw_ltr',
           'spread_to_pd_res_score_sa_ret_ew_le', 'spread_to_pd_res_score_sa_ret_sw_le', 'spread_to_pd_res_score_sa_ret_mw_le',
           'spread_to_pd_res_score_sa_ret_ew_ltr', 'spread_to_pd_res_score_sa_ret_sw_ltr', 'spread_to_pd_res_score_sa_ret_mw_ltr',
           'value_reg_richness_score_ret_ew_le', 'value_reg_richness_score_ret_sw_le', 'value_reg_richness_score_ret_mw_le',
           'value_reg_richness_score_ret_ew_ltr', 'value_reg_richness_score_ret_sw_ltr', 'value_reg_richness_score_ret_mw_ltr',
           'value_reg_richness_score_sa_ret_ew_le', 'value_reg_richness_score_sa_ret_sw_le', 'value_reg_richness_score_sa_ret_mw_le',
           'value_reg_richness_score_sa_ret_ew_ltr', 'value_reg_richness_score_sa_ret_sw_ltr', 'value_reg_richness_score_sa_ret_mw_ltr',
           'value_score_ret_ew_le', 'value_score_ret_sw_le', 'value_score_ret_mw_le',
           'value_score_ret_ew_ltr', 'value_score_ret_sw_ltr', 'value_score_ret_mw_ltr',
           'value_score_sa_ret_ew_le', 'value_score_sa_ret_sw_le', 'value_score_sa_ret_mw_le',
           'value_score_sa_ret_ew_ltr', 'value_score_sa_ret_sw_ltr', 'value_score_sa_ret_mw_ltr',
           'combined_score_ret_ew_le', 'combined_score_ret_sw_le', 'combined_score_ret_mw_le',
           'combined_score_ret_ew_ltr', 'combined_score_ret_sw_ltr', 'combined_score_ret_mw_ltr',
           'combined_score_sa_ret_ew_le', 'combined_score_sa_ret_sw_le', 'combined_score_sa_ret_mw_le',
           'combined_score_sa_ret_ew_ltr', 'combined_score_sa_ret_sw_ltr', 'combined_score_sa_ret_mw_ltr',
           'benchmark_ret_le', 'benchmark_ret_ltr']

df_final_ret = df_final.groupby(['DATE'])[COL_RET].sum()
df_final_ret.to_csv('strategy_portfolio_return.csv')
