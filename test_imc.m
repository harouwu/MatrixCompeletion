k = 50;
d1 = 50;
d2 = 50;
n1 = 100;
n2 = 100;
P = zeros(20, 1);
for p=0:19
    P(p+1) = n1*n2/20*p;
end
P(1) = 100;

r_imc = zeros(20, 1);
r_pg = zeros(20, 1);
for p=0:19
    r_imc(p+1) = do_test_imc(k, d1, d2, n1, n2, P(p+1));
    r_pg(p+1) = PGMC(n1, n2, P(p+1));
end

%%
a = P / double(n1*n2);

hold on
plot(a, r_imc, '-rd');
plot(a, r_pg, '-bx');
xlabel('Sparsity') % x-axis label
ylabel('Relative Error(%)') % y-axis label
title('IMC vs PGD matrix completion')
legend('IMC','PGD')