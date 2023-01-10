clear all, close all

x = gpml_randn(0.8, 20, 1); % 20 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1); % 20 noisy training targets
xs = linspace(-3, 3, 61)'; % 61 test inputs

meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1); % set parameters arbitrarily
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y); % set parameters by optimising log marginal likelihood

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs); % make predictions for gp

f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)]; % 2*std gives 95% decision regions
fill([xs; flip(xs,1)], f, [7 7 7]/8) % filL(X,Y,C) fills polygon defined by X,Y with C
hold on; plot(xs, mu); plot(x, y, '+') % plot mean function and training data