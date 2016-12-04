function [O] = gen_sparse(Y, Omega, n, m, p)

ii = zeros(p,1);
jj = zeros(p,1);
b = zeros(p,1);
for ij=1:p
    i = floor((Omega(ij)-1)/m)+1;
    ii(ij) = i;
    j = Omega(ij) - (i-1)*m;
    jj(ij) = j;
    b(ij) = Y(i, j);
end

O = sparse(ii,jj,b,n,m);

end
