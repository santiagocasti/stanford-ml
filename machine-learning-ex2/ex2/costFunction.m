function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%fprintf("X[%d,%d]\n", rows(X), columns(X));
%fprintf("y[%d,%d]\n", rows(y), columns(y));
%fprintf("theta[%d,%d]\n", rows(theta), columns(theta));

% calculate the sigmoid function of X*theta
h = sigmoid(X * theta);

% calcualte all the errors
errors = (-y .* log(h)) - ((1 - y).*log(1 - h));

% sum the errors and divide them by m
J = sum(errors) / m;

for i = 1:columns(X)
	% calculate the errors
	errors = (h - y) .* X(:,i); 
	% set the gradient
	grad(i) = sum(errors) / m;
endfor

% =============================================================

end
