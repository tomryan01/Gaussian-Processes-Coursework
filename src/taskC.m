clear all, close all

load('cw1a.mat') % load data for task (a)

xs = linspace(-3, 3, 101)'; % test data linspace

figure;

meanfunc = @meanZero;
covfunc = @covPeriodic; 
likfunc = @likGauss; 

hyp = struct('mean', [], 'cov', [-1 -0.7 0], 'lik', 0); % set parameters arbitrarily
[optHyp fX] = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

disp(fX)

[mu s2] = gp(optHyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs); % make predictions for gp

f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)]; % 2*std gives 95% decision regions

hold on;
name = "Trained Periodic Gaussian Process for cw1a.mat data";
title({
    name 
    sprintf("initial values: hyp.cov=[%.2f, %.2f, %.2f], hyp.lik = %.2f", hyp.cov(1), hyp.cov(2), hyp.cov(3), hyp.lik)
    sprintf("optimised values: hyp.cov=[%.2f, %.2f, %.2f], hyp.lik = %.2f", optHyp.cov(1), optHyp.cov(2), hyp.cov(3), optHyp.lik)
});
fill([xs; flip(xs,1)], f, [7 7 7]/8) % filL(X,Y,C) fills polygon defined by X,Y with C
plot(xs, mu); plot(x, y, '+'); xlabel('-3 < x < 3'); ylabel('f(x)') % plot mean function and training data