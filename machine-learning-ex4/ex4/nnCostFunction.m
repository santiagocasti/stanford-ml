function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%fprintf('Theta1[%d,%d]\n', rows(Theta1), columns(Theta1)); % 25x401
%fprintf('Theta2[%d,%d]\n', rows(Theta2), columns(Theta2)); % 10x26
% fprintf('X[%d,%d]\n', rows(X), columns(X)); % 5000x400
% fprintf('y[%d,%d]\n', rows(y), columns(y)); % 5000x1
% m = 5000;
vectorY = zeros(m,num_labels); % 
for i = 1:m
	vectorY(i,y(i)) = 1;
endfor
% fprintf('vectorY[%d,%d]\n', rows(vectorY), columns(vectorY)); % 5000x10
%disp(y(400:410));
%disp(vectorY(400:410,:));

% add a0(1) = 1
a1 = [ones(m,1) X];
% fprintf('a1[%d,%d]\n', rows(a1), columns(a1)); % 5000x401


z2 = Theta1 * (a1.');
% fprintf('z2[%d,%d]\n', rows(z2), columns(z2)); % 25x5000
a2 = sigmoid(z2);
% fprintf('a2[%d,%d]\n', rows(a2), columns(a2)); % 25x5000
% add a0(2) = 1 
a2 = [ones(1,m); a2];
% fprintf('a2[%d,%d]\n', rows(a2), columns(a2)); % 26x5000

z3 = Theta2 * a2;
% fprintf('z3[%d,%d]\n', rows(z3), columns(z3)); % 10x5000
a3 = sigmoid(z3);
% fprintf('a3[%d,%d]\n', rows(a3), columns(a3)); % 10x5000
%disp(a3);
a3 = a3.'; % a3 now is 5000x10
% m = 5000
% num_labels = K = 10
oneVector = ones(rows(vectorY),columns(vectorY));
tempJ = - vectorY.*log(a3) - (oneVector - vectorY).*log(oneVector - a3);
% fprintf('val[%d,%d]\n', rows(tempJ), columns(tempJ)); % 5000x10
tempJ = sum(tempJ,2);
tempJ = sum(tempJ,1);
tempJ = tempJ / m;
% disp(tempJ);

J = tempJ;


%fprintf('Theta1[%d,%d]\n', rows(Theta1), columns(Theta1)); % 25x401
%fprintf('Theta2[%d,%d]\n', rows(Theta2), columns(Theta2)); % 10x26
% disp(input_layer_size); % 400
% disp(hidden_layer_size); % 25

% regularization values calculation
theta1Sum = (Theta1(:,2:input_layer_size+1).^2);
theta1Sum = sum(theta1Sum, 2);
theta1Sum = sum(theta1Sum, 1);

theta2Sum = (Theta2(:,2:hidden_layer_size+1).^2);
theta2Sum = sum(theta2Sum, 2);
theta2Sum = sum(theta2Sum, 1);

regularizationTotal = (lambda * ( theta1Sum + theta2Sum )) / (2*m);

J = J + regularizationTotal;




% backpropagation

delta3 = a3 - vectorY;
% fprintf('delta3[%d,%d]\n', rows(delta3), columns(delta3)); % 5000x10

delta2 = (Theta2.') * (delta3.') .* (a2 .* (1 - a2));
% fprintf('delta2[%d,%d]\n', rows(delta2), columns(delta2)); % 26x5000
delta2 = delta2(2:end,:);
% fprintf('delta2[%d,%d]\n', rows(delta2), columns(delta2)); % 25x5000


a1_small = a1(:,2:end);
% fprintf('a1_small[%d,%d]\n', rows(a1_small), columns(a1_small)); % 5000x400
a1_small = a1_small.';
% fprintf('a1_small[%d,%d]\n', rows(a1_small), columns(a1_small)); % 400x5000

% a2_small = a2(2:end,:);
% fprintf('a2_small[%d,%d]\n', rows(a2_small), columns(a2_small)); % 25x5000

tempValue1 = delta2 * (a1);
% fprintf('tempValue1[%d,%d]\n', rows(tempValue1), columns(tempValue1)); % 25x400
tempValue2 = (delta3.') * (a2.');
% fprintf('tempValue2[%d,%d]\n', rows(tempValue2), columns(tempValue2)); % 10x25

% fprintf('Theta1_grad[%d,%d]\n', rows(Theta1_grad), columns(Theta1_grad)); % 25x401
Theta1_grad = tempValue1 / m;
% Theta1_grad = [zeros(rows(Theta1_grad),1) Theta1_grad];
% fprintf('Theta1_grad[%d,%d]\n', rows(Theta1_grad), columns(Theta1_grad)); % 25x401
% fprintf('Theta2_grad[%d,%d]\n', rows(Theta2_grad), columns(Theta2_grad)); % 10x26
Theta2_grad = tempValue2 / m;
% fprintf('Theta2_grad[%d,%d]\n', rows(Theta2_grad), columns(Theta2_grad)); % 10x26


% regularizing the gradient
temp1 = Theta1 * lambda / m;
temp1(:,1) = 0;
Theta1_grad = Theta1_grad + temp1;

temp2 = Theta2 * lambda / m;
temp2(:,1) = 0;
Theta2_grad = Theta2_grad + temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
