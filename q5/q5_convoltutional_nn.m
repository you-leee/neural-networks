# We have a convolutional neural network for images of 5 by 5 pixels. 
# In this network, each hidden unit is connected to a different 4 x 4 region of the input image:
# Top-left, top-right, bottom-left, bottom-right
# Because it's a convolutional network, the weights (connection strengths) are the same for all hidden units
# The hidden units are logistic. The network has no biases
# For the training case with that "3" input image, what is the output of each of the four hidden units?

W = [1,1,1,0;
     0,0,1,0;
     1,1,1,0;
     0,0,1,0];
     
p = [0,1,1,1,0;
     0,0,0,1,0;
     0,1,1,1,0;
     0,0,0,1,0;
     0,1,1,1,0];
     
function l = logit(z)
  l = 1/(1 + exp(-z));
endfunction

logit(sum(sum(p(1:4, 1:4) .* W)))
logit(sum(sum(p(1:4, 2:5) .* W)))
logit(sum(sum(p(2:5, 1:4) .* W)))
logit(sum(sum(p(2:5, 2:5) .* W)))
