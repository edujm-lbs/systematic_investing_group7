import pandas as pd

from carry import calculate_carry_signals
from momentum import calculate_momentum_signals
from quality import quality_calc
from utils import calculate_portfolio_weights, calculate_portfolio_returns
from value import calculate_value_signals


# Include any relevant field used for your analysis - avoid having such a large DataFrame
REL_COL = ['DATE', 'ISIN', 'AMOUNT_OUTSTANDING', 'RET_EOM', 'SPREAD_yield',
           'return_excess_by_duration', 'datadate', 'HSICCD', 'dlc', 'dltt',
           'mib', 'upstk', 'che', 'mv', 'dt', 'gp', 'at', 'ret_var_movstd_yrl',
           'mkvalt', 'DURATION', 'TMT', 'N_SP']

DIR = 'bondret_data17e.sas7bdat'

df_data = pd.read_sas(DIR).loc[:,REL_COL]
df_data = df_data[df_data['datadate'].notna()]

df_data['sector'] = df_data['HSICCD'].astype(str).str[0]
df_data['dts'] = df_data['DURATION'] * df_data['SPREAD_yield']
df_data['TMT_2'] = df_data['TMT'] ** 2

l_df = []
for date in df_data.DATE.sort_values().unique()[13:]:
    print(date)
    # For momentum signal calculation we require trailing dates.
    df_dt = df_data[df_data.DATE <= date].copy()
    # 1. Calculation of signals (should it be raw or output directly Z-score?)
    df_dt_m = calculate_momentum_signals(df_dt, date)
    # Carry
    df_dt_m_c = calculate_carry_signals(df_dt_m)
    # Quality
    df_dt_m_c_q = quality_calc(df_dt_m_c)
    # Value
    df_dt_m_c_q_v = calculate_value_signals(df_dt_m_c_q)
    # 2. Combine all Style combined Z-Scores on to one
    df_dt_m_c_q['combined_score'] = \
        df_dt_m_c_q[['momentum_score', 'carry_score', 'quality_score', 'value_score']].mean(axis=1)
    df_dt_m_c_q['combined_score_sa'] = \
        df_dt_m_c_q[['momentum_score_sa', 'carry_score_sa', 'quality_score_sa', 'value_score_sa']].mean(axis=1)
    # 3. Fn to implement logic to pick top ranked scores and provide weights for all bonds in a given month
    df_dt_m_c_q_w = calculate_portfolio_weights(df_dt_m_c_q)
    # 4. Calculate market cap weight for benchmark calculations later on
    df_dt_m_c_q_w['benchmark_wght'] = \
        df_dt_m_c_q_w['AMOUNT_OUTSTANDING'] / df_dt_m_c_q_w['AMOUNT_OUTSTANDING'].sum()
    l_df.append(df_dt_m_c_q_w)

df_final = pd.concat(l_df)

df_final = calculate_portfolio_returns(df_final)
df_final.to_csv('final_df.csv')

COL_RET = ['mom_spread_6_score_ret_ew', 'mom_spread_6_score_ret_sw', 'mom_spread_6_score_ret_mw',
           'mom_spread_6_score_sa_ret_ew', 'mom_spread_6_score_sa_ret_sw', 'mom_spread_6_score_sa_ret_mw',
           'mom_spread_12_m_1_score_ret_ew', 'mom_spread_12_m_1_score_ret_sw',
           'mom_spread_12_m_1_score_sa_ret_ew', 'mom_spread_12_m_1_score_sa_ret_sw',
           'mom_spread_12_m_1_score_ret_mw', 'momentum_score_ret_ew', 'momentum_score_ret_sw',
           'mom_spread_12_m_1_score_sa_ret_mw', 'momentum_score_sa_ret_ew', 'momentum_score_sa_ret_sw',
           'momentum_score_ret_mw', 'carry_score_ret_ew', 'carry_score_ret_sw', 'carry_score_ret_mw',
           'momentum_score_sa_ret_mw', 'carry_score_sa_ret_ew', 'carry_score_sa_ret_sw', 'carry_score_sa_ret_mw',
           'leverage_z_ret_ew',	'leverage_z_ret_sw', 'leverage_z_ret_mw', 
           'leverage_z_sa_ret_ew', 'leverage_z_sa_ret_sw', 'leverage_z_sa_ret_mw',
           'profit_z_ret_ew', 'profit_z_ret_sw', 'profit_z_ret_mw',
           'profit_z_sa_ret_ew', 'profit_z_sa_ret_sw', 'profit_z_sa_ret_mw',
           'quality_score_ret_ew', 'quality_score_ret_sw', 'quality_score_ret_mw',
           'quality_score_sa_ret_ew', 'quality_score_sa_ret_sw', 'quality_score_sa_ret_mw',
           'spread_to_pd_res_score_ew', 'spread_to_pd_res_score_sw', 'spread_to_pd_res_score_mw',
           'spread_to_pd_res_score_sa_ew', 'spread_to_pd_res_score_sa_sw', 'spread_to_pd_res_score_sa_mw',
           'value_reg_richness_score_ew', 'value_reg_richness_score_sw', 'value_reg_richness_score_mw',
           'value_reg_richness_score_sa_ew', 'value_reg_richness_score_sa_sw', 'value_reg_richness_score_sa_mw',
           'value_score_ret_ew', 'value_score_ret_sw', 'value_score_ret_mw',
           'value_score_sa_ret_ew', 'value_score_sa_ret_sw', 'value_score_sa_ret_mw',
           'combined_score_ret_ew', 'combined_score_ret_sw', 'combined_score_ret_mw',
           'combined_score_sa_ret_ew', 'combined_score_sa_ret_sw', 'combined_score_sa_ret_mw',
           'benchmark_ret']

df_final_ret = df_final.groupby(['DATE'])[COL_RET].sum()

df_final_ret.to_csv('final_df_ret.csv')