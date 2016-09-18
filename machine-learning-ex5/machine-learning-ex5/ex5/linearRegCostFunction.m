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

% Compute cost
e = hypothesis(X, theta) - y; % m x 1
J = (e' * e) / (2 * m);

% Add the regularization
reg = (theta(2:end)' * theta(2:end)) * lambda / (2 * m);
J = J + reg;

% Compute the gradient
% nx1 = nxm * mx1
grad = X' * e / m; % Compute the unregularized gradient
grad_reg = lambda * theta(2:end) / m;
grad(2:end) = grad(2:end) + grad_reg;
% for j = 1:m
% for j = 1:length(theta)
%     % 1x1 = 1xm * mx1
%     grad(j) = X(:,j)' * e / m;
%     % nx1 = 1xn * mx1
%     grad = grad + (X(j,:) * e / m);
%     grad(1) = grad(1) + (X(j,1) * e / m); 
%     grad(2:end) = grad(2:end) + (X(j,2:end) * e / m); % Compute the unregularized gradient
%     grad_reg = lambda * theta(j) / m; % Compute regularization for gradient
%     grad(j) = grad(j) + grad_reg;
% end

% =========================================================================

grad = grad(:);

end

function h = hypothesis(X, theta)
    h = X * theta;
end
