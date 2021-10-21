%{
See "Example01.m" and "Example02.m" for detailed comments
%}

clear all;
close all;
clc;

%% Generate the data
t = 0:0.1:10; t=t';
n = t/0.1; 
GaussNoise = 0.5*randn(length(t),1);

b = [13 5 6];
a = [40 2 3 6 9];

ColoredNoise = filter(b,a,GaussNoise);

TrendedNoise = ColoredNoise + sin(2*t);

%% Compare the residule
% Use the digital filter
FFT_TrendedNoise = fft(TrendedNoise);
plot(abs(FFT_TrendedNoise));
index = 4;
len = length(FFT_TrendedNoise);
A = 2*abs(FFT_TrendedNoise(index))/len;

FFT_TrendedNoise = fft(TrendedNoise,1024*10);
plot(abs(FFT_TrendedNoise));
index = 326;
len = length(FFT_TrendedNoise);
w = 2*pi*index/len;

phi = angle(FFT_TrendedNoise(index));

DetrendedNoise = TrendedNoise - A*cos(w*n + phi);

% Use the difference operator
TrendedNoise_diff = SeasonalDiff(TrendedNoise,31);

% Plot detrended residuals
figure;
plot(n(32:end),TrendedNoise_diff,'g','linewidth',2);
hold on;
plot(n,ColoredNoise,'b--','linewidth',2);
hold on;
plot(n,DetrendedNoise,'r','linewidth',2);
axis([0 n(end) -2 2]);
legend('ARIMA (first order diff)','Real Value','ARMA-SIN');
xlabel('Time Step','fontsize',14);
% ylabel('Acceleration (m/s^2)','fontsize',14);
set(gca,'fontsize',14);

%% Use the ARIMA and ARMA-SIN to predict
predLen = 20;

%% ARIMA
data = TrendedNoise;
len = length(data);

dataModel = arima('Constant',0,'Seasonality',31,'MALags',1,'ARLags',4);
dataFit = estimate(dataModel,data(1:len-predLen));

YY = [];
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
data = DetrendedNoise;
len = length(data);

dataModel = arima(4,0,1);
dataFit = estimate(dataModel,data(1:len-predLen));

YY = [];
for i = len-predLen:len
    yy = forecast(dataFit,1,'Y0',data(1:i));
    YY = [YY; yy];
end

NonstationaryTrend = (TrendedNoise - DetrendedNoise);   % which is "A*cos(w*n + phi)" above ...
% one can also use "A*cos(w*n + phi)" to predict future trend, they are the same

hold on;
plot(len - predLen:len, [data(len-predLen)+NonstationaryTrend(len - predLen); YY(1:end-1) + NonstationaryTrend(len - predLen + 1:len)],'m','linewidth',2);

err_armasin = (TrendedNoise(len - predLen+1:len)) - (YY(1:end-1) + NonstationaryTrend(len - predLen+1:len));

display(['MSE ARMA-SIN: ' num2str(GetMSE(err_armasin))]);

axis([0 n(end) -2 2]);
legend('Real Value (past)','Real Value (future)','ARIMA (first order diff)','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);