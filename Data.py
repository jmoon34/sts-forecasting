import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns

import collections

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts




with open("item_1.csv") as f:
    demand = [x.strip().split(",")[-1] for x in f.readlines()[1:]]


demand_dates = np.arange('2013-01-01', '2018-01-01', dtype="datetime64[D]")
demand = np.array(demand).astype(np.float32)
demand = demand
num_forecast_steps = 365 # Forecast the final year by using previous data
demand_loc = mdates.MonthLocator((6, 12))
demand_fmt = mdates.DateFormatter('%b %Y')
demand_training_data = demand[:-num_forecast_steps]

colors = sns.color_palette()
c1, c2 = colors[0], colors[1]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)


print(len(demand_dates), len(demand))

ax.plot(demand_dates[:-num_forecast_steps], demand[:-num_forecast_steps], lw=0.5, label="training data")
ax.xaxis.set_major_locator(demand_loc)
ax.xaxis.set_major_formatter(demand_fmt)
ax.set_ylabel("Daily demand")
ax.set_xlabel("Date")
fig.suptitle("Daily Demand of Store Item", fontsize=15)
fig.autofmt_xdate()
plt.show()

