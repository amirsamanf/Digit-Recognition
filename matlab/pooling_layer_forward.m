function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = input.batch_size;
    
    output.data = zeros([h_out, w_out, c, batch_size]);
    for batch = 1:batch_size
        dat = reshape(input.data(:,batch), input.height, input.width, input.channel);
        for ch = 1:c
            row = 1;
            im = dat(:,:,ch);
            %Zero pad dat
            paddedImage = padarray(im, [pad pad], 'both');
            %Convolution
            for i = pad+1:stride:size(paddedImage, 1) - pad
                col = 1;
                for j = pad+1:stride:size(paddedImage, 2) - pad
                    max = 0;
                    for kRow = 0:k-1
                        for kCol = 0:k-1
                            if paddedImage(i+kRow,j+kCol) > max
                                max = paddedImage(i+kRow,j+kCol);
                            end
                        end
                    end
                    output.data(row,col,ch,batch) = max;
                    col = col+1;
                end
                row = row+1;
            end
        end
    end
    output.data = reshape(output.data, h_out*w_out*c, batch_size);
end

