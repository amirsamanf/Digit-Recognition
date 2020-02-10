% Q3.3 Classifying 8 Real World Examples (numbers from 0 - 7)

% get layers
layers = get_lenet();

%load model
load lenet.mat
 
layers{1}.batch_size = 1;
predictions = zeros(1,8);
for number = 1:8
    % Real world examples are 0,1,2,3,4,5,6,7
    fileName = sprintf('%d.png', number-1);
    img = imread(fileName);
    img = rgb2gray(img)';
    img = reshape(img, 28*28,1);
    [output,P] = convnet_forward(params, layers, img);
    [~,ind] = max(P);
    predictions(number) = ind-1;
end

fprintf('Predictions are below (should be 0 1 2 3 4 5 6 7):\n')
disp(predictions);














