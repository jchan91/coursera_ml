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

for i = 1:m
    h = hypothesis(Theta1, Theta2, X(i,:));
    posCase = -convertY(y(i), num_labels)' * log(h); 
    negCase = -(1 - convertY(y(i), num_labels))' * log(1 - h);
    J = J + posCase + negCase;
end
J = J / m;

% Regularize
J = J + regularize(Theta1, lambda, m) + regularize(Theta2, lambda, m);

% prediction
%activation2 = sigmoid(computeActivationLayer(Theta1, X));
%output = sigmoid(computeActivationLayer(Theta2, activation2));


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

for t = 1:m
    % Step 1: Compute activation layers (feed forward)
    a1 = X(t,:)';
    [a2, z2] = computeActivationLayer(Theta1, a1);
    [a3, z3] = computeActivationLayer(Theta2, a2);
    
    assert(size(a2,2) == 1); % Ensure column
    assert(size(z2,2) == 1);
    assert(size(a1,2) == 1);
    
    % Step 2: Compute first back propogation error, d3
    d3 = a3 - convertY(y(t), num_labels);
    
    % Step 3: Compute hidden layer (l=2) back propagation, d2
    d2 = Theta2' * d3 .* sigmoidGradient([1; z2]);
    
    % Step 4: Accumulate the gradients of all the examples into one matrix for
    % each layer. Note that these Deltas are the same dim as Theta1, Theta2
    % Note that we need to append 1 to the activation layers again, because
    % we have to incorporate *how much error the bias weights contribute*
    Theta1_grad = Theta1_grad + (d2(2:end) * [1; a1;]');
    Theta2_grad = Theta2_grad + (d3 * [1; a2;]');
end

% Step 5: Normalize Delta1 and Delta2 by num_samples

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda * Theta1_grad(:,2:end) / m);
% if lambda > 1
%     Theta2_grad(:,2:end)
% end
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda * Theta2_grad(:,2:end) / m);
% if lambda > 1
%     Theta2_grad(:,2:end)
%     pause;
% end
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function h = hypothesis(theta1, theta2, x)
    z2 = sigmoid(theta1 * [1 x]');
    h = sigmoid(theta2 * [1; z2;]);
end

function yVec = convertY(y, num_labels)
    % Returns a column logical array with idx given by scalar value set to
    % true
    yVec = zeros(num_labels, 1);
    yVec(y) = 1;
end

function r = regularize(theta, lambda, num_samples)
    r = lambda / (2 * num_samples) * sum(sum(theta(:,2:end).^2)); % Square all the parameters of theta, and sum them
end

function [a, z] = computeActivationLayer(theta, x)
    % Exact same logic as "hypothesis"
    z = theta * [1; x;];
    a = sigmoid(z);
end