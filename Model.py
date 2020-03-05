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

from Data import *
from Plot import *


num_days_per_month = np.array(
    [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # 2013
     [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # 2014
     [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # 2015
     [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], # 2016, leap year
     [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]] # 2017
)

def build_model(observed_time_series):
    day_of_week_effect = sts.Seasonal(num_seasons=7,
                                      observed_time_series=observed_time_series,
                                      name='day_of_week_effect')
    month_of_year_effect = sts.Seasonal(num_seasons=12,
                                        num_steps_per_season=num_days_per_month,
                                        observed_time_series=observed_time_series,
                                        name='month_of_year_effect')
    autoregressive = sts.Autoregressive(
        order=1,
        observed_time_series=observed_time_series,
        name='autoregressive')
    model = sts.Sum([day_of_week_effect, month_of_year_effect, autoregressive],
                    observed_time_series=observed_time_series)
    return model


tf.reset_default_graph()
demand_model = build_model(demand_training_data)

# Build variational loss function and surrogate posteriors 'qs'
with tf.variable_scope('sts_elbo', reuse=tf.AUTO_REUSE):
    elbo_loss, variational_posteriors = tfp.sts.build_factored_variational_loss(demand_model, demand_training_data)
train_vi = tf.train.AdamOptimizer(0.1).minimize(elbo_loss)

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 201
num_variational_steps = int(num_variational_steps)

# Run optimization and draw samples from surrogate posteriors
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_variational_steps):
        _, elbo_ = sess.run((train_vi, elbo_loss))
        if i % 20 == 0:
            print("step {} - ELBO {}".format(i, elbo_))

    q_samples_demand_ = sess.run({k: q.sample(50) for k, q in variational_posteriors.items()})

print("Inferred parameters:")
for param in demand_model.parameters:
    print("{}: {} +- {}".format(param.name,
                                np.mean(q_samples_demand_[param.name], axis=0),
                                np.std(q_samples_demand_[param.name], axis=0)))




# Forecasting, IMPORTANT PART

demand_forecast_dist = tfp.sts.forecast(
    model=demand_model,
    observed_time_series=demand_training_data,
    parameter_samples=q_samples_demand_,
    num_steps_forecast=num_forecast_steps)

num_samples = 10

with tf.Session() as sess:
    (demand_forecast_mean, demand_forecast_scale, demand_forecast_samples) = sess.run(
        (demand_forecast_dist.mean()[..., 0],
         demand_forecast_dist.stddev()[..., 0],
         demand_forecast_dist.sample(num_samples)[..., 0]))

fig, ax = plot_forecast(demand_dates, demand,
                        demand_forecast_mean,
                        demand_forecast_scale,
                        demand_forecast_samples,
                        title="Item demand forecast",
                        x_locator=demand_loc, x_formatter=demand_fmt)

fig.tight_layout()
plt.show()

component_dists = sts.decompose_by_component(
    demand_model,
    observed_time_series=demand_training_data,
    parameter_samples=q_samples_demand_)

forecast_component_dists = sts.decompose_forecast_by_component(
    demand_model,
    forecast_dist=demand_forecast_dist,
    parameter_samples=q_samples_demand_)


with tf.Session() as sess:
    demand_component_means_, demand_component_stddevs_ = sess.run(
        [{k.name: c.mean() for k, c in component_dists.items()},
         {k.name: c.stddev() for k, c in component_dists.items()}])

    [demand_forecast_component_means_, demand_forecast_component_stddevs_] = sess.run(
        [{k.name: c.mean() for k, c in forecast_component_dists.items()},
         {k.name: c.stddev() for k, c in forecast_component_dists.items()}])

# Concatenate the training data with forecasts for plotting
component_with_forecast_means_ = collections.OrderedDict()
component_with_forecast_stddevs_ = collections.OrderedDict()
for k in demand_component_means_.keys():
    component_with_forecast_means_[k] = np.concatenate([
        demand_component_means_[k],
        demand_forecast_component_means_[k]], axis=-1)
    component_with_forecast_stddevs_[k] = np.concatenate([
        demand_component_stddevs_[k],
        demand_forecast_component_stddevs_[k]], axis=-1)

fig, axes = plot_components(
    demand_dates,
    component_with_forecast_means_,
    component_with_forecast_stddevs_,
    x_locator=demand_loc, x_formatter=demand_fmt)

for ax in axes.values():
    ax.axvline(demand_dates[-num_forecast_steps], linestyle="--", color='red')

plt.show()
