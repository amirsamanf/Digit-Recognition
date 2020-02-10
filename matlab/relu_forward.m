function [output] = relu_forward(input)

    output.height = input.height;
    output.width = input.width;
    output.channel = input.channel;
    output.batch_size = input.batch_size;

    output.data = max(input.data, 0);

end
