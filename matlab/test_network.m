%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
l = size(xtest, 2);
C = zeros(10,10);
predictionVect = zeros(1, l);
for i=1:100:l
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [~,ind] = max(P);
    predictionVect(i:i+99) = ind;
end
for i = 1:l
   test = ytest(i);
   pred = predictionVect(i);
   C(test, pred) = C(test, pred) + 1;
end

fprintf('Confusion matrix for test set below:\n')
disp(C);