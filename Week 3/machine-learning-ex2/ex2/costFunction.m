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

hypothesis = zeros(m,1);
size(hypothesis)
##for i=1:m
##  hypothesis(i) = (sigmoid(X(i,:)*theta));
##endfor

hypothesis = (sigmoid(X*theta));

J  = -(1/m)*(sum(y .*log(hypothesis) + (1-y) .* log(1- hypothesis)));


deviation_y = hypothesis - y;
size(deviation_y);
##descent = zeros(length(theta),1);
##
##for j = 1:length(theta)
##  x = X(:,j);
##  grad(j) = (1/m)*sum(deviation_y .*x);
##endfor

grad = (1/m)* deviation_y'*X;





% =============================================================

end