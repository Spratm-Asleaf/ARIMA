clear all;
close all;
clc;

% AirPassengers, Monthly Airline Passenger Numbers, 1949-1960.
% This dataset is available anywhere, e.g., online, R built-in, Python built-in.
data = [
 112 118 132 129 121 135 148 148 136 119 104 118
 115 126 141 135 125 149 170 170 158 133 114 140
 145 150 178 163 172 178 199 199 184 162 146 166
 171 180 193 181 183 218 230 242 209 191 172 194
 196 196 236 235 229 243 264 272 237 211 180 201
 204 188 235 227 234 264 302 293 259 229 203 229
 242 233 267 269 270 315 364 347 312 274 237 278
 284 277 317 313 318 374 413 405 355 306 271 306
 315 301 356 348 355 422 465 467 404 347 305 336
 340 318 362 348 363 435 491 505 404 359 310 337
 360 342 406 396 420 472 548 559 463 407 362 405
 417 391 419 461 472 535 622 606 508 461 390 432   
]';

data = data(:);         % convert to a column vector
len = length(data);
n = 1:len; n = n';      % index

log_data =log(data);    % log transform


%% Spectral analysis for log-data
log_data_fft = fft(log_data);
w = linspace(0,2,len);
plot(w, abs(log_data_fft), 'linewidth', 2);
xlabel('Discrete Frequency ($\times \pi$)','fontsize',14,'interpreter','latex');
set(gca,'fontsize',14);
axis([0 w(end) 0 20]);


%% De-trend
% Use the digital filter to de-trend
[ order , Wn] = ellipord (0.04 , 0.05 , 1 , 10);
[ b , a] = ellip(order , 1, 10, Wn, 'high');
DetrendedData = filtfilt(b, a, log_data);

% Use the difference operator
DetrendedData_diff = diff(log_data);

% Plot detrended residuals
figure;
plot(n(2:end),DetrendedData_diff,'g','linewidth',2);
axis([0 n(end) -0.5 0.5]);
xlabel('Time Step','fontsize',14);
ylabel('First Order Diff','fontsize',14);
set(gca,'fontsize',14);
figure;
plot(n,DetrendedData,'r','linewidth',2);
axis([0 n(end) -0.5 0.5]);
ylabel('ARMA-SIN','fontsize',14);
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);

%% Extract Periodic Patterns (i.e., Seasonal Patterns)
% FFT pattern of detrended series
% We can see the low-frequency trend has been removed already
% But then, the following discrete frequency points need to be de-season-ed
DetrendedData_FFT = fft(DetrendedData,len);
figure;
w = linspace(0,2,len);
plot(w,abs(abs(DetrendedData_FFT)),'r','linewidth',2);
xlabel('Discrete Frequency ($\times \pi$)','fontsize',14,'interpreter','latex');
set(gca,'fontsize',14);
W = [0.04196 0.1678 0.3357 0.5035 0.6713 0.8392]';      % frequency points where large-valued outliers happen in the frequency domain

%% De-season
DetrendedDeseasonedData = DetrendedData;
for i = 1:length(W)
    a = W(i) - 0.015;
    b = W(i) + 0.015;
    a_below = max(a - 0.015, 0.015);    % a_below cannot be smaller than 0; use 0.015 instead of 0 for safety purpose
    b_upper = min(b + 0.015, 0.985);    % b_upper cannot be larger than 1; use 0.985 instead of 1 for safety purpose
    Fpass1 = a_below;       % First Passband Frequency
    Fstop1 = a;             % First Stopband Frequency
    Fstop2 = b;             % Second Stopband Frequency
    Fpass2 = b_upper;       % Second Passband Frequency
    Apass1 = 0.01;          % First Passband Ripple (dB)
    Astop  = 5;             % Stopband Attenuation (dB)
    Apass2 = 0.01;          % Second Passband Ripple (dB)
    match  = 'both';        % Band to match exactly

    % Construct an FDESIGN object and call its ELLIP method.
    h  = fdesign.bandstop(Fpass1, Fstop1, Fstop2, Fpass2, Apass1, Astop, ...
                          Apass2);
    Hd = design(h, 'ellip', 'MatchExactly', match);
    [b,a] = tf(Hd);

    % De-season one-by-one
    DetrendedDeseasonedData = filtfilt(b,a,DetrendedDeseasonedData);
end

%% Fit the season part
% First, see the pattern in the frequency domain
figure;
w = linspace(0,2,len);
plot(w,abs(fft(DetrendedData - DetrendedDeseasonedData)),'r','linewidth',2);
xlabel('Discrete Frequency ($\times \pi$)','fontsize',14,'interpreter','latex');
set(gca,'fontsize',14);
W = [0.04196 0.1678 0.3357 0.5035 0.6713 0.8392]';  % frequency points where large-valued outliers happen in the frequency domain
Indices = [4 13 25 37 49 61];   % Indices = round(W*len/2 + 1), because W starts from zero while Indices start from 1
                                % Indices indicate positions (from 1 to len) where W happens
                                
Season = 0;                     % Extract Seasonal component
for i = 2:6                     % The first large value in W corresponds to the low-frequency trend, not seasonal components
    A = abs(DetrendedData_FFT(Indices(i)));
    phi = angle(DetrendedData_FFT(Indices(i)));
    A = 2*A/len;
    Season = Season + A*cos(pi*W(i)*n + phi);
end

% Do note that "Season" is the estimate to "DetrendedData - DetrendedDeseasonedData"

% FFT pattern of de-trended and de-season-ed series
figure;
w = linspace(0,2,512); w= w';
plot(w,abs(fft(DetrendedData,512)),'b',w,abs(fft(DetrendedDeseasonedData,512)),'r','linewidth',2);
xlabel('Discrete Frequency ($\times \pi$)','fontsize',14,'interpreter','latex');
set(gca,'fontsize',14);
axis([0 w(end) 0 20]);


%% Find proper orders for WSS ARMA part
% SARIMA
DetrendedDeseasonedData_diff = SeasonalDiff(DetrendedData_diff, 12);
figure;
autocorr(diff(DetrendedDeseasonedData_diff),50);
figure;
parcorr(diff(DetrendedDeseasonedData_diff),50);                     % Therefore, use ARMA(3, 1) for WSS part

% ARMA-SIN
figure;
autocorr((DetrendedDeseasonedData),50);
figure;
parcorr((DetrendedDeseasonedData),50);                              % Therefore, use ARMA(2, 1) for WSS part                                  

%% Use ARIMA and ARMA-SIN to predict
predLen = 20;       % to predict 20 points in the future

%% Using SARIMA
data = log_data;                % feed original log-transformed data
len = length(data);

dataModel = arima('D',1,'Seasonality',12,'MALags',[],'ARLags',1); % But significant test (t test) tells us that there should not exist MA
dataFit = estimate(dataModel,data(1:len-predLen));           % Use past data for training

YY = [];                        % Predicted value
for i = len-predLen:len
    yy = forecast(dataFit,1,'Y0',data(1:len-predLen));       % long-term forecasting       
    %yy = forecast(dataFit,1,'Y0',data(1:i));                % if using rolling (one-step ahead short-term) forecasting, use "data(1:i)" instead
    YY = [YY; yy];
end

figure;
plot(1:len - predLen, exp(data(1:len-predLen)),'b','linewidth',2);                   % historic data
hold on;
plot(len - predLen:len, exp(data(len-predLen:len)),'b--','linewidth',2);             % real future data
hold on;
plot(len - predLen:len, exp([data(len-predLen); YY(1:end-1)]),'g','linewidth',2);    % forecasted future data using SARIMA  

err_arima = data(len - predLen+1:len) - YY(1:end-1);                                 % MSE

display(['MSE ARIMA: ' num2str(GetMSE(err_arima))]);


%% Using ARMA-SIN
data = DetrendedDeseasonedData;          % Stationary ARMA part after de-trending and de-seasoning
                                         % Remember to add back the seasonal part and the trend part for final forecasting

len = length(data);

dataModel = arima('Constant',0,'MALags',2,'ARLags',1);
dataFit = estimate(dataModel,data(1:len-predLen));

YY = [];                                 % Predicted values
for i = len-predLen:len
    yy = forecast(dataFit,1,'Y0',data(1:len-predLen));                      % long-term forecasting         
    %yy = forecast(dataFit,1,'Y0',data(1:i));                               % if using rolling (one-step ahead short-term) forecasting, use "data(1:i)" instead
    YY = [YY; yy];
end

NonstationaryTrend = log_data - Season - DetrendedDeseasonedData;           % Extract Trend (it also roughly equals to "log_data - DetrendedData",
                                                                            % because "Season" approximates "DetrendedData - DetrendedDeseasonedData")

hold on;
p_coef = PolynomialFit(n, NonstationaryTrend, 1);                           % Use polynomial fitting to treat 
                                                                            % non-stationary trend part

plot(len - predLen:len, ... 
    exp([data(len-predLen) + Season(len-predLen) + NonstationaryTrend(len - predLen); ...                         % Historic real record
    YY(1:end-1) + Season(len - predLen+1:len) + (p_coef(1) * n(len - predLen+1:len) + p_coef(2))]),...            % Predicted values: WSS + Season + Trend
    'm','linewidth',2);

err_armasin = log_data(len - predLen+1:len) - (Season(len - predLen+1:len) + YY(1:end-1) + (p_coef(1) * n(len - predLen+1:len) + p_coef(2)));

display(['MSE ARMA-SIN: ' num2str(GetMSE(err_armasin))]);

%axis([0 n(end) 4.5 6.5]);
legend('Real Value (past)','Real Value (future)','ARIMA (first order diff)','ARMA-SIN');
xlabel('Time Step','fontsize',14);
set(gca,'fontsize',14);