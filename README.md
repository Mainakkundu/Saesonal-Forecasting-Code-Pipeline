# Saesonal-Forecasting-Code-Pipeline
In retail sector few NOOS(never oot of stock items) exhibit a peculiar type of seasonality and mostly it was event base , and multiplicative in nature.Machine Learning models are very diffcilut to estimate that Multiplicative pattern YoY.
So this arhictecture will build on Box-Jenkinson methodology,it works as follows:
1) Identfy the series is Stationary or Non-Stationary 
2) Tag the treatment require for nake it a stationary time series 
3) Then it will find for Yearly seasonality Exist or Not 
4) Then do the simulation using auto_arima() -- both the parameters-p,d,q & P,D,Q
5) Then it will do the forecasting 
