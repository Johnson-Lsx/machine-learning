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
%==========================part 1 ===============================
%原本的X是5000 * 400，我们需要加上bais，之后a1变成5000 * 401
a1 = [ones(m,1) X];
%进行前向传播
z2 = a1 * Theta1';
a2 = 1 ./ ( 1 + e.^(-z2) );
a2 = [ones( size(a2,1),1 ) a2];
z3 = a2 * Theta2';
a3 = 1 ./ ( 1 + e.^(-z3) );
%计算损失函数
%h的维度是5000 * 10，y的维度是5000 * 1
h = a3;
%因此对于每个y的值，我们需要把将其变成只包含0、1的向量
Y = zeros(m,num_labels);
for i = 1 : m
  Y(i,y(i)) = 1;
end
%计算J
for i = 1 : m
  J += -(log(h(i,:)) * Y(i,:)' + log(1 - h(i,:)) * (1 - Y(i,:))');
end
%在前面的基础上加上正则化项
Theta1L2 = 0;
Theta2L2 = 0;
for i = 1 : size(Theta1,1)
  Theta1L2 += Theta1(i,:) * Theta1(i,:)' - Theta1(i,1)^2;
end

for i = 1 : size(Theta2,1)
  Theta2L2 += Theta2(i,:) * Theta2(i,:)' - Theta2(i,1)^2;
end

J = J / m + lambda / (2 * m) * (Theta1L2 + Theta2L2);
%============================part 2==============================
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for i = 1 : m
  %feedfowward
  a_1 = [1 ; X(i,:)'];
  z_2 = Theta1 * a_1;
  a_2 = [1 ; sigmoid(z_2)];
  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);
  %compute the delta terms
  delta_3 = a_3 - Y(i,:)';
  delta_2 = Theta2' * delta_3 .* [1 ; sigmoidGradient(z_2)];
  delta_2 = delta_2(2:end);
  Delta_2 += delta_3 * a_2';
  Delta_1 += delta_2 * a_1';
end
Theta1_grad = Delta_1 / m + (lambda / m) * Theta1;
Theta1_grad(:,1) -= (lambda / m) * Theta1(:,1);
Theta2_grad = Delta_2 / m + (lambda / m) * Theta2;
Theta2_grad(:,1) -= (lambda / m) * Theta2(:,1);











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
