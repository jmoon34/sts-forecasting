The following code is based on the article: https://medium.com/tensorflow/structural-time-series-modeling-in-tensorflow-probability-344edac24083

This was before I started learning about machine learning and the Tensorflow framework, so the core of the code is from the article.  I wrote code to import data for the demand of an item of a store from Kaggle, and took a small portion of it to do forecasting using Tensorflow Probability.  The model took into account day of the week and month of the year and autoregressive components as its parameters.


## Demand Data

![Demand Data](/Diagrams/Data_Graph.png?raw=true "Demand Data")

## Forecast

![Forecast Graph](/Diagrams/Forecast_Graph.png?raw=true "Forecast")

Dotted orange line: Mean of the forecast
Shaded orange region: +- 2 standard deviations of the forecast
Solid blue line: Actual data


## Components

![Components Graph](/Diagrams/Components_Graph.png?raw=true "Components")

This graph breaks down the contribution of each parameter that contributed to the forecasting model.



## Output:
step 0 - ELBO 5855.15185546875<br>
step 20 - ELBO 4566.06640625<br>
step 40 - ELBO 4407.68896484375<br>
step 60 - ELBO 4366.677734375<br>
step 80 - ELBO 4348.19873046875<br>
step 100 - ELBO 4345.94482421875<br>
step 120 - ELBO 4343.2529296875<br>
step 140 - ELBO 4340.3154296875<br>
step 160 - ELBO 4342.306640625<br>
step 180 - ELBO 4336.9990234375<br>
step 200 - ELBO 4337.12060546875<br>

*Inferred parameters:*
observation_noise_scale: 4.243539333343506 +- 0.08322174847126007<br>
day_of_week_effect/_drift_scale: 0.13437619805335999 +- 0.0468049980700016<br>
month_of_year_effect/_drift_scale: 3.5961546897888184 +- 0.43267887830734253<br>
autoregressive/_coefficients: [0.1635988] +- [0.3852669]<br>
autoregressive/_level_scale: 0.3180446922779083 +- 0.2202991545200348<br>


