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

% Feedforward

A_1 = [ones(m,1), X];
Z_2 = A_1 * Theta1';
A_2 = sigmoid(Z_2); % a matrix of vectors in hidden layer 1, each row correspends to an example

A_2 = [ones(m,1),A_2];
Z_3 = A_2 * Theta2';
A_3= sigmoid(Z_3); % each row is the output vector corresponding to each an example

theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);
Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
for i = 1:m
    % initialize current y to a k-dimentianl vector (k = num_labels)
    curr_y = zeros(num_labels,1);
    curr_y(y(i)) = 1;
    
    % Extract every example's correspending vector in each layer.
    a_3 = A_3(i,:)';
    a_2 = A_2(i,:)';
    z_2 = Z_2(i,:)';
    a_1 = A_1(i,:)';
    z_1 = X(i,:)';
    
    % Update cost function
    J = J  +  -1/m * sum(curr_y.*log(a_3) + (1-curr_y).*log(1-a_3));

    % Back-propagation
    delta_3 = a_3 - curr_y; % column vector
    delta_2 = (Theta2(:,2:end)' * delta_3).* sigmoidGradient(z_2);
    delta_1 = (Theta1(:,2:end)' * delta_2).* sigmoidGradient(z_1);
    
    Delta_2 = Delta_2 +  delta_3 * a_2'; % Not multiplying dot product!
    
    Delta_1 = Delta_1 +  delta_2 * a_1' ;
   
end
Theta2_grad = Delta_2/m + lambda/m*Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda/m*Theta2(:,1);
Theta1_grad = Delta_1/m + lambda/m*Theta1;
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda/m*Theta1(:,1);

J = J + lambda/(2*m)*(sum(theta1(:).^2) + sum(theta2(:).^2)); % Regularization










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
