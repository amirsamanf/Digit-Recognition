function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 

output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = input.batch_size;
output.data = zeros([h_out, w_out, num, batch_size]);
ker = reshape(param.w, k,k,c,num);
padding = floor(k / 2);
for batch = 1:batch_size
    dat = reshape(input.data(:,batch), input.height, input.width, input.channel);
    %Zero pad dat
    paddedImage = padarray(dat, [pad pad], 'both');
    for n = 1:num
        row = 1;
        %Convolution
        for i = padding+1:stride:size(paddedImage, 1) - padding
            col = 1;
            for j = padding+1:stride:size(paddedImage, 2) - padding
                val = 0;
                for kRow = -padding:padding
                    for kCol = -padding:padding
                        for kDep = 1:c
                            val = val + paddedImage(i+kRow,j+kCol,kDep) * ker(kRow+padding+1,kCol+padding+1,kDep,n);
                        end
                    end
                end
                output.data(row,col,n,batch) = val;
                col = col+1;
            end
            row = row+1;
        end
        output.data(:,:,n,batch) = output.data(:,:,n,batch) + param.b(n);
    end
end
output.data = reshape(output.data, h_out*w_out*num, batch_size);

end

