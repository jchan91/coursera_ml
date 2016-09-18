function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m, n] = size(X); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute cost
positiveCase = -y' * log(hypothesis(X, theta));
negativeCase = -(1 - y)' * log(1 - hypothesis(X, theta));
regularization = 0.5 * lambda * (theta(2:n)' * theta(2:n));
J = (positiveCase + negativeCase + regularization) / m;

% Compute gradient
grad(1) = (hypothesis(X, theta) - y)' * X(:,1);
for j = 2:n
    gradient = (hypothesis(X, theta) - y)' * X(:,j);
    reg = lambda * theta(j);
    
    grad(j) = gradient + reg;
end
grad = grad / m;

% =============================================================

end

function [h] = hypothesis(X, theta)

h = sigmoid(X * theta);

end