clear all;
close all;
clc;

%% Generate simulation data
t = 0:0.5:100; t=t';
n = t/0.5;                                  % Sampling time is 0.5

GaussNoise = randn(length(t),1);            % Gaussian white series

b = [13 5 6];                               % Eq. (3)
a = [40 2 3 6 9];

ColoredNoise = filter(b,a,GaussNoise);      % Eq. (4)
                                            % WSS time series of interest

TrendedNoise = ColoredNoise + 0.1*t;        % Linear trend "0.1t"


%% Compare the detrended residule
% Use the digital filter
[ order , Wn] = ellipord (0.25 , 0.2 , 1 , 10);
[ b , a] = ellip(order , 1, 10, Wn, 'high');
DetrendedNoise = filtfilt(b,a,TrendedNoise);
% Find proper orders for WSS ARMA part
figure;
autocorr(diff(DetrendedNoise),50);
figure;
parcorr(diff(DetrendedNoise),50);
% FFT pattern of detrended series and original trended series
figure;
plot(1:512,abs(fft(DetrendedNoise,512)),'b',1:512,abs(fft(TrendedNoise,512)),'r--','linewidth',2);

% Use the difference operator
DetrendedNoise_diff = diff(TrendedNoise);

% Plot residuals
figure;
plot(n(2:end),DetrendedNoise_diff,'g','linewidth',2);
hold on;
plot(n,ColoredNoise,'b--','linewidth',2);
hold on;
plot(n,DetrendedNoise,'r','linewidth',2);
axis([0 n(end) -2 2]);
legend('ARIMA (first order diff)','Real Value','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);

%% Use the ARIMA and ARMA-SIN to predict
predLen = 20;       % to predict 20 points in the future

%% ARIMA
data = TrendedNoise;
len = length(data);

dataModel = arima(4,1,2);       % Use best arima model (p = 4 and q = 2)
dataFit = estimate(dataModel,data(1:len-predLen));

YY = [];                        % Predicted value
for i = len-predLen:len
    yy = forecast(dataFit,1,'Y0',data(1:i));
    YY = [YY; yy];
end

figure;
plot(1:len - predLen, data(1:len-predLen),'b','linewidth',2);
hold on;
plot(len - predLen:len, data(len-predLen:len),'b--','linewidth',2);
hold on;
plot(len - predLen:len, [data(len-predLen); YY(1:end-1)],'g','linewidth',2);

err_arima = data(len - predLen+1:len) - YY(1:end-1);

display(['MSE ARIMA: ' num2str(GetMSE(err_arima))]);


%% ARMA-SIN
data = DetrendedNoise;          % Stationary ARMA part
len = length(data);

dataModel = arima(4,0,1);       % Other orders like (8, 0, 2) also triable
dataFit = estimate(dataModel,data(1:len-predLen));

YY = [];                        % Predicted values
for i = len-predLen:len
    yy = forecast(dataFit,1,'Y0',data(1:i));
    YY = [YY; yy];
end

NonstationaryTrend = TrendedNoise - DetrendedNoise;

hold on;
p_coef = PolynomialFit(t, NonstationaryTrend, 1);                           % Use polynomial fitting to treat 
                                                                            % non-stationary trend part

plot(len - predLen:len, ... 
    [data(len-predLen) + NonstationaryTrend(len - predLen); ...             % History real record
    YY(1:end-1) + p_coef(1) * t(len - predLen+1:len) + p_coef(2)],...       % Predicted values
    'm','linewidth',2);

err_armasin = TrendedNoise(len - predLen+1:len) - (YY(1:end-1) + p_coef(1) * t(len - predLen+1:len) + p_coef(2));

display(['MSE ARMA-SIN: ' num2str(GetMSE(err_armasin))]);

axis([0 n(end) -2 11]);
legend('Real Value (past)','Real Value (future)','ARIMA (first order diff)','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);