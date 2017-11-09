function Z = partition(rbm_w)
  num_hidden = rows(rbm_w);
  num_visible = columns(rbm_w);
  
  % Create all possible states
  hidden_states = create_states(num_hidden);
  visible_states = create_states(num_visible);
  
  Z = 0;
  
  for i=1:rows(hidden_states)
    for j=1:rows(visible_states)
      E = hidden_states(i,:) * rbm_w * visible_states(j,:)';
      Z = Z + exp(-E);

    end    
  end
end
  