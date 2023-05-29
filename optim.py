import pandas as pd
import numpy as np
from scipy.optimize import linprog
from itertools import repeat

###########################################################################
# We did not end up optimizing portfolio given the limited time resources #
###########################################################################

def get_optim_ports(data, max_wgt, sector_bound, credit_bound, duration_bound, dts_bound):
  """
  this will return a dataframe of the optimal portfolio, along weight the market weighted
  benchmark for each date in your dataframe
  this function assumes the score you want to optimize is in a column called 'score'
  the objective function is to maximize the inner product of the weights and the score
  s.t.
  x > 0%
  x <= max_wgt%, max_wgt should be entered as a decimal
  sum(x) == 1
  portfolio sector must be +/- sector_bound% relative to benchmark, sector_bound should be entered as a decimal
  portfolio credit rating must be +/- credit_bound% relative to benchmark, credit_bound should be entered as a decimal
  portfolio duration must be +/- duration_bound years relative to benchmark, duration_bound should be entered as an integer
  you need to ensure that your sector column is called HSICCD which is what the SAS script called it
  example of how ot calculate data['HSICCD'] = data['HSICCD'].astype(str).str[0]
  you also need to ensure you have DTS already calculated which is simply data["DTS"] = data["DURATION"] * data["SPREAD_yield"]
  
  params: data; dataframe of all holdings
    max_wgt; the maximum weight for a given securty, entered as a decimal
    sector_bound; the maximum deviation from each sector relative to the benchmark, entered as a decimal
    credit_bound; the maximum deviation from each credit rating relative to the benchmark, entered as a decimal
    duration_bound; the maximum deviation from the benchmark duration, entered as an integer
  
  returns: a dataframe with the optimal weight for your portfolio for all dates as well as the market-value weights for the bmk
    all columns returned will be needed for the plot sas script
  """
  # create an empty df that we'll append things onto later
  master_holdings_df = pd.DataFrame()

  # get the dates through which we iterate
  dates = data.DATE.unique()

  # we'll need a column of 1s for our pivot
  data['one'] = 1

  # loop through each date and optimize the portfolio
  for my_date in dates:
    print(my_date)
    # just take the columns we care about
    this_months_holdings_df = data[data['DATE'] == my_date]
    this_months_holdings_df = this_months_holdings_df[["ISIN", "sector", "score", "AMOUNT_OUTSTANDING", "YEAR_MONTH_d", "DATE", "SPREAD_yield", "lead_TOT_RET", "lead_EXCESS_RET", "MATURITY_yrs", "DURATION", "T_Spread", "YIELD", "TMT", "R_MR",  "PRICE_EOM", "DTS"]]
    this_months_holdings_df["MARKET_WEIGHT_B"] = this_months_holdings_df["AMOUNT_OUTSTANDING"] / sum(this_months_holdings_df["AMOUNT_OUTSTANDING"])

    # SECTOR
    # lhs
    sector_lhs = this_months_holdings_df[["ISIN","sector","one"]].pivot(index='sector', columns ='CUSIP').fillna(0).sort_index().to_numpy()

    # rhs
    # we're basing this off the bmk weights
    sector_rhs = this_months_holdings_df.groupby("sector")["MARKET_WEIGHT_B"].sum().sort_index().to_numpy()

    # bounds
    # sector_bound = 0.05
    sector_rhs_u = sector_rhs + sector_bound
    sector_rhs_d = sector_rhs - sector_bound

    # CREDIT RATING
    # lhs
    credit_lhs = this_months_holdings_df[["ISIN","R_MR","one"]].pivot(index='R_MR', columns ='CUSIP').fillna(0).sort_index().to_numpy()

    # rhs
    # we're basing this off the bmk weights
    credit_rhs = this_months_holdings_df.groupby("R_MR")["MARKET_WEIGHT_B"].sum().sort_index().to_numpy()

    # bounds
    # credit_bound = 0.05
    credit_rhs_u = credit_rhs + credit_bound
    credit_rhs_d = credit_rhs - credit_bound

    # DURATION
    # lhs
    duration = this_months_holdings_df["DURATION"].to_numpy()
    duration_lhs = np.transpose(np.atleast_2d(duration).T)

    # rhs
    # based on bmk
    duration_rhs = this_months_holdings_df[["ISIN","MARKET_WEIGHT_B","DURATION"]]
    duration_rhs["contrib"] = duration_rhs["MARKET_WEIGHT_B"] * duration_rhs["DURATION"]
    duration_rhs = duration_rhs["contrib"].sum()

    # bounds
    # duration_bound = 2
    duration_rhs_u = np.array([duration_rhs + duration_bound])
    duration_rhs_d = np.array([duration_rhs - duration_bound])

    # DTS
    # lhs
    dts = this_months_holdings_df["DTS"].to_numpy()
    dts_lhs = np.transpose(np.atleast_2d(dts).T)
    
    # rhs
    dts_rhs = this_months_holdings_df[["ISIN","MARKET_WEIGHT_B","DTS"]]
    dts_rhs["contrib"] = dts_rhs["MARKET_WEIGHT_B"] * dts_rhs["DTS"]
    dts_rhs = dts_rhs["contrib"].sum()
    
    # bounds
    dts_rhs_u = np.array([dts_rhs + dts_bound])
    dts_rhs_d = np.array([dts_rhs - dts_bound])
    
    # weights must sum to 1
    # lhs
    sum_to_one_lhs = this_months_holdings_df["one"].to_numpy()
    sum_to_one_lhs = np.transpose(np.atleast_2d(sum_to_one_lhs).T)

    # rhs
    # note that we need to expand the dimensions for the optimization function
    sum_to_one_rhs = np.expand_dims(np.array([1]), axis=1)

    # get the coefficients for the objective function
    # note that we're multiplying everything by -1 because we're doing minimization
    scores = this_months_holdings_df['score'] * -1

    # establish security bound
    min_wgt = 0
    # max_wgt = 0.05

    security_bounds = list(repeat((min_wgt, max_wgt), sum_to_one_lhs.shape[1]))

    # setup the inequality matrices
    lhs_ineq = np.concatenate((sector_lhs, 
                    sector_lhs * -1, # for the lower bound
                    credit_lhs,
                    credit_lhs * -1, # for the lower bound
                    duration_lhs,
                    duration_lhs * -1, # for the lower bound
                    dts_lhs,
                    dts_lhs * -1), axis=0)

    rhs_ineq = np.concatenate((sector_rhs_u,
                               sector_rhs_d * -1,
                               credit_rhs_u,
                               credit_rhs_d * -1,
                               duration_rhs_u,
                               duration_rhs_u * -1,
                               dts_rhs_u,
                               dts_rhs_d * -1), axis=0)

    # the lhs must have 2 dimensions
    rhs_ineq = np.expand_dims(rhs_ineq, axis=1)

    # run the optimization
    opt = linprog(c=scores, 
                  A_ub=lhs_ineq, 
                  b_ub=rhs_ineq, 
                  A_eq=sum_to_one_lhs, 
                  b_eq=sum_to_one_rhs, 
                  bounds=security_bounds, 
                  method="revised simplex")

    # add the weights we got to our dataframe
    this_months_holdings_df["optim_wgt"] = opt.x

    # append this months holdings to our master set
    master_holdings_df = pd.concat([master_holdings_df, this_months_holdings_df])
    
return(master_holdings_df)
