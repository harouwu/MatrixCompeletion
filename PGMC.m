function [relerr] = PGMC(n, m, p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   Solving the question:
%   1/2 \sum_{(i,j) \in \Omega} (Y_{ij} - B_{ij})^2 + \lambda ||B||_{tr}

if nargin < 1, n = 100;		end
if nargin < 2, m = 100;		end
if nargin < 3, p = 5000;		end


max_iter = 20;
lambda = 1;
% generate random data
fprintf('Generating random data...');
[Y, B, O, Omega] = gen_rand(n, m, p);

for iter=1:max_iter
    fprintf('Iter %d. \n', iter);
    PB = gen_sparse(B, Omega, n, m, p);
    PBP = B - PB;
    B = MST(O + PBP, lambda);
end

relerr = norm(Y - B,'fro')^2 / norm(Y,'fro')^2 * 100;
fprintf('RelErr = %e\n',relerr);

end

function [Y, B, O, Omega] = gen_rand(n, m, p)
Y = randn(n,m);
B = randn(n,m);
Omega = randsample(n*m,p);
O = gen_sparse(Y, Omega, n, m, p);
end


