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


hypothesis = zeros(m,1);
size(hypothesis);
##for i=1:m
##  hypothesis(i) = (sigmoid(X(i,:)*theta));
##endfor
hypothesis = (sigmoid(X*theta));

J  = -(1/m)*(sum(y .*log(hypothesis) + (1-y) .* log(1- hypothesis))) + ...
  (lambda/(2*m)).*(sum(theta.^2) - theta(1).^2) ;


deviation_y = hypothesis - y;
size(deviation_y);
##descent = zeros(length(theta),1);

##for j = 1:length(theta)
##  x = X(:,j);
##  if(j==1)
##    grad(j) = (1/m)*sum(deviation_y .*x);
##  else
##    grad(j) = (1/m)*sum(deviation_y .*x) + (lambda/m).*theta(j);
##  endif
##endfor

theta(1)=0

grad = (1/m)* deviation_y'*X + (lambda/m).*theta;

% =============================================================

end
