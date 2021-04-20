function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

iter_C = 0.01;
iter_sigma = 0.01;
opt_C = iter_C;
opt_sigma = iter_sigma;

model = svmTrain(X, y, iter_C, @(x1, x2)gaussianKernel(x1, x2, iter_sigma));
predictions = svmPredict(model, Xval);
error = mean(double(predictions ~= yval));
min_error = error;

for i = 1:7
    iter_sigma = 0.01;
    for j = 1:7
        model = svmTrain(X, y, iter_C, @(x1, x2)gaussianKernel(x1, x2, iter_sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error <= min_error
            min_error = error;
            opt_C = iter_C;
            opt_sigma = iter_sigma;
        end
        
        if mod(j,2) == 1
            iter_sigma = iter_sigma * 3;
        else
            iter_sigma = iter_sigma * 10 / 3;
        end
    end
    
    if mod(i,2) == 1
        iter_C = iter_C * 3;
    else
        iter_C = iter_C * 10 / 3;
    end
end

C = opt_C
sigma = opt_sigma




% =========================================================================

end
