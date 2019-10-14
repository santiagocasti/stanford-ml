function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% I will use the previous cost calculation and add the new term
[J, grad] = costFunction(theta, X, y);

% For the cost function we should add: (sum(theta^2) * lambda/2m) for theta from 1 to n
J = J + ((lambda * sum(theta(2:rows(theta)).^2))/(2*m));

% For the gradient we should add: (lambda * theta_j / m) for theta from 1 to n
for j = 2:rows(theta)
	grad(j) = grad(j) + (lambda * theta(j)) / m;
endfor


% =============================================================

end
