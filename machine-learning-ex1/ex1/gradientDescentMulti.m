function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
num_features = size(X,2);
new_theta = zeros(num_features,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % for the number of features present
    for feature = 1:num_features
        
        % obtain the parameter X^(i)
        X_i = X(:,feature);
        
        % calculate the derivative term of the cost function
        derivativeTerm1 = sum(((X * theta) - y) .* X_i) / m;

        % assign the new value of theta as the old one minus alpha times the derivative term
        new_theta(feature) = theta(feature) - (alpha * derivativeTerm1);
    end

    % assign all the new theta at the same time
    theta = new_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end