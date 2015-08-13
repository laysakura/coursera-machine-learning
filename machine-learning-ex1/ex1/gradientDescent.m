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

    % $\theta_j := \theta_j$- \alpha \Sigma_{i=1}^{m} ( \theta X^{(i)} - y^{(i)} ) X^(i)$ ($1 <= j <= n$)
    n = length(theta);
    new_theta = zeros(n, 1);
    for j = 1:n
        partial_sum = 0.0;
        for i = 1:m
            partial_sum += (1.0 / m) *  (X(i, :) * theta - y(i)) * X(i, j);
        end
        new_theta(j) = theta(j) - alpha * partial_sum
    end
    theta = new_theta;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
