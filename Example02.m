%{
See "Example01.m" for detailed comments
%}

clear all;
close all;
clc;

%% Generate the data
t = 0:0.1:50; t=t';
n = t/0.1;
GaussNoise = 0.5*randn(length(t),1);

b = [13 5 6];
a = [40 2 3 6 9];

ColoredNoise = filter(b,a,GaussNoise);

TrendedNoise = ColoredNoise + sin(5*t);

Fpass1 = 0.158;     % First Passband Frequency
Fstop1 = 0.16;      % First Stopband Frequency
Fstop2 = 0.165;     % Second Stopband Frequency
Fpass2 = 0.168;     % Second Passband Frequency
Apass1 = 1;         % First Passband Ripple (dB)
Astop  = 20;        % Stopband Attenuation (dB)
Apass2 = 1;         % Second Passband Ripple (dB)
match  = 'both';    % Band to match exactly

% Construct an FDESIGN object and call its ELLIP method.
h  = fdesign.bandstop(Fpass1, Fstop1, Fstop2, Fpass2, Apass1, Astop, ...
                      Apass2);
Hd = design(h, 'ellip', 'MatchExactly', match);
[b,a] = tf(Hd);

DetrendedNoise = filtfilt(b,a,TrendedNoise);

% Use the difference operator
TrendedNoise_diff = SeasonalDiff(TrendedNoise,12);

% WSS part (detrended residuals)
figure;
plot(n(13:end),TrendedNoise_diff,'g','linewidth',2);
hold on;
plot(n,ColoredNoise,'b--','linewidth',2);
hold on;
plot(n,DetrendedNoise,'r','linewidth',2);

axis([0 n(end) -2 2]);
legend('ARIMA (first order diff)','Real Value','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);

%% Use the ARIMA and ARMA-SIN to predict
predLen = 20;

%% SARIMA
data = TrendedNoise;
len = length(data);

dataModel = arima('Constant',0,'D',1,'Seasonality',12,'MALags',1,'ARLags',1:4);
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

NonstationaryTrend = (TrendedNoise - DetrendedNoise);           % The trend is a seasonal trend
                                                                % So using a sine to fit the seasonal trend
                                                                
NonstationaryTrend_FFT = fft(NonstationaryTrend);
A = 2*abs(NonstationaryTrend_FFT(41))/501;
w = (41/501)*2*pi;      % the peak is located at the 41th point among 501 points
                        % 0.1 is the sampling time
phi = angle(NonstationaryTrend_FFT(41));
% Hence, A*cos(w*n + phi) is a good approximation of "NonstationaryTrend"
% This is by the Fourier series approximation theory
hold on;
plot(len - predLen:len, ...
    [data(len-predLen)+NonstationaryTrend(len - predLen); ...
    YY(1:end-1) + A*cos(w*n(len - predLen + 1:len) + phi)],...
    'm','linewidth',2);

err_armasin = TrendedNoise(len - predLen+1:len) - (YY(1:end-1) + A*cos(w*n(len - predLen + 1:len) + phi));

display(['MSE ARMA-SIN: ' num2str(GetMSE(err_armasin))]);

axis([0 n(end) -2 2]);
legend('Real Value (past)','Real Value (future)','ARIMA (first order diff)','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);