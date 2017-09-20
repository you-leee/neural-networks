batchsize = 100;  % Mini-batch size.
learning_rate = 0.1;  % Learning rate; default = 0.1.
momentum = 0.9;  % Momentum; default = 0.9.
numhid1 = 50;  % Dimensionality of embedding space; default = 50.
numhid2 = 200;  % Number of units in hidden layer; default = 200.

[train_input, train_target, valid_input, valid_target, ...
  test_input, test_target, vocab] = load_data(batchsize);
[numwords, batchsize, numbatches] = size(train_input); 
vocab_size = size(vocab, 2);


word_embedding_weights = zeros(vocab_size, numhid1);
embed_to_hid_weights = zeros(numwords * numhid1, numhid2);
hid_to_output_weights = zeros(numhid2, vocab_size);
hid_bias = zeros(numhid2, 1);
output_bias = zeros(vocab_size, 1);
      
[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
        fprop(valid_input, word_embedding_weights, embed_to_hid_weights,...
              hid_to_output_weights, hid_bias, output_bias);
datasetsize = size(valid_input, 2);

expansion_matrix = eye(vocab_size);
tiny = exp(-30);
expanded_valid_target = expansion_matrix(:, valid_target);
CE = -sum(sum(...
        expanded_valid_target .* log(output_layer_state + tiny))) /datasetsize