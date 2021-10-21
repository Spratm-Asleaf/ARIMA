function [ mse ] = GetMSE( x )
%GETMSE Summary of this function goes here
%   Detailed explanation goes here
    mse = sum(x.^2)/length(x);
end

