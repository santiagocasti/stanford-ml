function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % get the corresponding column for each featuer
    X1 = X(:,1); % all ones
    X2 = X(:,2);

    % calculate the derivative term for the first feature
    derivativeTerm1 = sum(((X * theta) - y) .* X1 ) ./ m;
    % calculate theta0 based on the first feature
    theta0 = theta(1) - (alpha .* derivativeTerm1);

    % calculate the derivetive term for the seconf feautre
    derivativeTerm2 = sum(((X * theta) - y) .* X2 ) ./ m;
    % calculate theta1 based on the second feature
    theta1 = theta(2) - (alpha .* derivativeTerm2);

    theta = [theta0; theta1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
