clear all, close all

load('cw1e.mat'); % load data for task (a)

x1 = linspace(-4,4,100); x2 = linspace(-4,4,100);
xs = apxGrid('expand',{x1',x2'});

% E1
figure;

meanfunc = @meanZero;
covfunc = @covSEard; 
likfunc = @likGauss; 

hyp = struct('mean', [], 'cov', [1, 1, 1], 'lik', 0);
[optHyp fX] = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
disp("Single covSEard: ");
disp(fX(length(fX)));

[mu s2] = gp(optHyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs); % make predictions for gp
top = reshape(mu + 2*sqrt(s2), 100, 100);
bottom = reshape(mu - 2*sqrt(s2), 100, 100);
mu = reshape(mu, 100, 100);

hold on;
subplot(1,2,1)
%mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
%mesh(x1, x2, mu');
%mesh(x1, x2, top');
%mesh(x1, x2, bottom');
mesh(x1, x2, (top - bottom)');
name = "Sum of covSEard: Difference Between 95% Confidence Bands";
title({
    name 
    sprintf("initial values: hyp.cov=[%.2f, %.2f, %.2f], hyp.lik = %.2f", hyp.cov(1), hyp.cov(2), hyp.cov(3), hyp.lik)
    sprintf("optimised values: hyp.cov=[%.2f, %.2f, %.2f], hyp.lik = %.2f", optHyp.cov(1), optHyp.cov(2), optHyp.cov(3), optHyp.lik)
});
hold off;

% E2

meanfunc = @meanZero;
covfunc = {@covSum, {@covSEard, @covSEard}};
likfunc = @likGauss; 

hyp = struct('mean', [], 'cov', [1, 1, 1, 1, 1, 1], 'lik', 0);
hyp.cov = 0.1*randn(6,1);
[optHyp fX] = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
%disp("Sum covSEard: ");
%disp(fX(length(fX)));

x1 = linspace(-4,4,100); x2 = linspace(-4,4,100);
xs = apxGrid('expand',{x1',x2'});

[mu s2] = gp(optHyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs); % make predictions for gp
top = reshape(mu + 2*sqrt(s2), 100, 100);
bottom = reshape(mu - 2*sqrt(s2), 100, 100);
mu = reshape(mu, 100, 100);

hold on;
%figure(f2);
subplot(1,2,2)

%mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));

%mesh(x1, x2, top');
%mesh(x1, x2, bottom');

%mesh(x1, x2, mu');

mesh(x1, x2, (top - bottom)');

name = "Sum of covSEard: Difference Between 95% Confidence Bands";
title({
    name 
    sprintf("initial values: hyp.cov=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f], hyp.lik = %.2f", hyp.cov(1), hyp.cov(2), hyp.cov(3), hyp.cov(4), hyp.cov(5), hyp.cov(6), hyp.lik)
    sprintf("optimised values: hyp.cov=[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f], hyp.lik = %.2f", optHyp.cov(1), optHyp.cov(2), optHyp.cov(3), optHyp.cov(4), optHyp.cov(5), optHyp.cov(6), optHyp.lik)
});
hold off;