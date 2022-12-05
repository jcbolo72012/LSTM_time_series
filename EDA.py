import pandas as pd
import collections
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import sys

dataframe = pd.read_csv('./static/BSR.csv',  engine='python')

# print(dataframe.head())
# asins = list(dataframe['PRODUCT_ASIN'])

# ctr = collections.Counter(asins)
# print(ctr)


# 'B08VD4Q7XW': 132
# 'B003WZIOHQ': 794

rslt_df = dataframe[dataframe['PRODUCT_ASIN'] == 'B003WZIOHQ'] 
rslt_df_2 = rslt_df[rslt_df['SELLER_ID'] == 'A308FMC4TWC7I2']
data = rslt_df_2[['TOP_100_RANK']].values.astype('float32')

# sellers_list = list(rslt_df['SELLER_ID'])
# ctr = collections.Counter(sellers_list)
# print(ctr)
# sys.exit()


# x = [datetime.datetime.strptime(d[0],'%Y-%m-%d').date() for d in rslt_df[['AS_OF_DATE']].values]


plt.plot(data)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.scatter(x,dataset)
# plt.gcf().autofmt_xdate()
plt.show()