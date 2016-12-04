function [ Blambda ] = MST( B, lambda )
%MST Summary of this function goes here
%   Detailed explanation goes here
[U,S,V] = svd(B);
S = max(S - lambda, 0);
Blambda = U*S*V';
end

