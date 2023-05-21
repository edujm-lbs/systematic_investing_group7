# import pandas as pd
# import numpy as np
# import os
import pandas as pd
import numpy as np
from utils import z_score

# # this assumes you're working on the wrds server and you've already run
# # the initial sample creation script
# # change this to your username
# username = "yourusername"

# # set working directory
# os.chdir(f"/home/lbs/{username}/E499_Group_Project/data/")

# def z_score(x):
#     """
#     used to calculate z-scores
#     """
#     return (x - np.mean(x)) / np.std(x)
  
# # name of the sample file
# sasdb = "bondret_data17e.sas7bdat"

# data = pd.read_sas(sasdb)

# # drop values where there's no datadate
# # this is what what was done in the example SAS script
# data = data[data['datadate'].notna()]

# # create sector
# # directly from SAS script
# data['sector'] = data['HSICCD'].astype(str).str[0]

################################################################################################################
# NOTES ON QUALITY
################################################################################################################
# https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/Common-Factors-in-Corporate-Bond-Returns.pdf
# we use leverage to measure quality
# to calculate leverage we use the same measure used in the above AQR paper
# (net debt + preferred stock + miniority interest) / (net debt + market value of equity)
# we found that taking dt (total debt) directly from the db results in ~70% coverage of the data
# to get higher coverage, when dt is NA, we instead take the sum of long term and short term debt
# when calculating factor returns, we found th results to be robust if we used dt or our revised version of total debt
# so in order to get better coverage, we'll use our revised calculation

# research in both this paper and the course textbook suggests that bonds from issuers of a higher quality
# tend to outperform on a risk-adjusted basis
# our findings were generally consistent with this, we calculated a Sharpe ratio of ~0.27 on a credit excess return basis
# in our best quality quintile portfolio
# compared with a Sharpe ratio of ~0.16 for the worst quality quintile portfolio
# from a raw outperformance perspective, the top quintile portfolio only outperformed the bottom
# by ~30 bps a year on a credit excess return basis
# we can see clear performance improvement on a risk adjusted basis
# marginal outperformance on a raw basis
# note that when calculating the quality score we formed sector neutral z-scores cross-sectionally

# we also evaluated measure of gross profitability (namely gross profit/total assets)
# and probability of default
# neither of these were as efficacious as just looking at leverage

# # first, replace all instances of short term and debt term debt that are missing with 0
# data['dlc'].fillna(0, inplace=True) # current debt
# data['dltt'].fillna(0, inplace=True) # long-term debt
# data['mib'].fillna(0, inplace=True) # minority interest
# data['upstk'].fillna(0, inplace=True) # preferred stock
# data['che'].fillna(0, inplace=True) # cash and cash equivalents

# # now calculate our revised measure of total debt
# data['total_debt'] = np.where(data.dt.isnull(), data.dlc + data.dltt, data.dt)

# # calculate leverage
# data['leverage'] = (data['total_debt'] + data['mib'] + data['upstk'] - data['che']) / (data['total_debt'] - data['che'] + data['mv'])

# # use our z-score function to calculate a sector-neutral z-score
# # this will be our measure of quality
# data['quality'] = data.groupby(['DATE','sector'])['leverage'].apply(lambda x: z_score(x))

def quality_calc(df):
    # first, replace all instances of short term and debt term debt that are missing with 0
    df['dlc'].fillna(0, inplace=True) # current debt
    df['dltt'].fillna(0, inplace=True) # long-term debt
    df['mib'].fillna(0, inplace=True) # minority interest
    df['upstk'].fillna(0, inplace=True) # preferred stock
    df['che'].fillna(0, inplace=True) # cash and cash equivalents

    # now calculate our revised measure of total debt
    df['total_debt'] = np.where(df.dt.isnull(), df.dlc + df.dltt, df.dt)

    # calculate leverage
    df['leverage'] = (df['total_debt'] + df['mib'] + df['upstk'] - df['che']) / (df['total_debt'] - df['che'] + df['mv'])

    # use our z-score function to calculate a sector-neutral z-score
    # this will be our measure of quality
    df['quality'] = df.groupby(['DATE','HSICCD'])['leverage'].apply(lambda x: z_score(x))
    
    return(df)
