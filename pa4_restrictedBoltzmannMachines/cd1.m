function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);

    hidden_probs = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_sample = sample_bernoulli(hidden_probs);
    
    pos_gradient = configuration_goodness_gradient(visible_data, hidden_sample);
    
    visible_sample = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden_sample));
    hidden_probs_2 = visible_state_to_hidden_probabilities(rbm_w, visible_sample);
    
    neg_gradient = configuration_goodness_gradient(visible_sample, hidden_probs_2);
    
    ret = pos_gradient - neg_gradient;
end
