layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;

layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
% imshow(img')
 
output = convnet_forward(params, layers, xtest(:, 1));
output_2 = reshape(output{2}.data, 24, 24, 20);
output_3 = reshape(output{3}.data, 24, 24, 20);

% Fill in your code here to plot the features.
for k=1:20
    % Layer 2 output
    figure(1)
    im1 = output_2(:,:,k)';
    im1 = uint8(255 * mat2gray(im1));
    subplot(4,5,k)
    imshow(im1)
    
    % Layer 3 output
    figure(2)
    im2 = output_3(:,:,k)';
    im2 = uint8(255 * mat2gray(im2));
    subplot(4,5,k)
    imshow(im2)
end



