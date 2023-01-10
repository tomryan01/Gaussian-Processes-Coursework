meanfunc = @meanZero;
covfunc = {@covProd, {@covPeriodic, @covSEiso}}; 
likfunc = @likGauss;

figure;

n = 200;
x = linspace(-5,5,200)';

hyp = struct('mean', [], 'cov', [-0.5 0 0 2 0], 'lik', 0);

K = feval(covfunc{:}, hyp.cov, x) + 1e-6*eye(200);
mu = feval(meanfunc, hyp.mean, x);

hold on;
subplot(1,2,1);
y = chol(K)'*randn(n, 1) + mu;
plot(x,y);

name = "Generated Gaussian Process 1";
title({
    name 
    sprintf("values: hyp.cov=[%.2f, %.2f, %.2f, %.2f, %.2f], hyp.lik = %.2f", hyp.cov(1), hyp.cov(2), hyp.cov(3), hyp.cov(4), hyp.cov(5), hyp.lik)
});

hold off;

hold on;
subplot(1,2,2);
y = chol(K)'*randn(n, 1) + mu;
plot(x,y);
name = "Generated Gaussian Process 2";
title({
    name 
    sprintf("values: hyp.cov=[%.2f, %.2f, %.2f, %.2f, %.2f], hyp.lik = %.2f", hyp.cov(1), hyp.cov(2), hyp.cov(3), hyp.cov(4), hyp.cov(5), hyp.lik)
});
hold off;