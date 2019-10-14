function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% fprintf('X[%d,%d]\n', rows(X), columns(X)); % 12x2
% fprintf('y[%d,%d]\n', rows(y), columns(y)); % 12x1
% fprintf('theta[%d,%d]\n', rows(theta), columns(theta)); % 2x1

h = X * theta;

temp = (h - y).^2;
temp = sum(temp,1) / (2*m);

reg = theta.^2;
reg = sum(reg(2:rows(reg)),1);
reg = reg * lambda / (2*m);

J = temp + reg;


temp = (h - y).';
temp = temp * X;
temp = temp / m;
% fprintf('temp[%d,%d]\n', rows(temp), columns(temp)); % 2x1

reg = (lambda * theta)/m;
reg(1) = 0;
reg = reg.';
% fprintf('reg[%d,%d]\n', rows(reg), columns(reg)); % 2x1

grad = temp + reg;

% =========================================================================

grad = grad(:);

end
