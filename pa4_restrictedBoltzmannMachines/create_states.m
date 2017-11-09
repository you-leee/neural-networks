function possible_states = create_states(n)
  possible_states = [];
  
  perms = 2^n;
  k = 2^(n-1);
  
  while(k >= 1)
    fill_vec = [ones(k, 1); zeros(k, 1)];
    
    possible_states = [possible_states, repmat(fill_vec, perms/k/2, 1)];
    
    k = k/2;
  endwhile
  
end
  