function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

output.data = zeros([n, k]);
output.height = layer.num;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

for i = 1:k
    output.data(:,i) = param.w' * input.data(:,i) + param.b';
end
