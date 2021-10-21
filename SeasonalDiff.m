function [ out ] = SeasonalDiff( x, L )
%SEASONALDIFF Summary of this function goes here
%   Detailed explanation goes here
    len = length(x);
    
    out = [];
    for i = L+1:len
        out = [out; x(i) - x(i-L)];
    end
end

