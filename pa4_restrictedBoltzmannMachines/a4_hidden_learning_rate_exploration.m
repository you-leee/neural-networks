hidden_to_class_learning_rates1 = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1];

for lr = hidden_to_class_learning_rates1
  a4_main(300, .02, lr, 1000)
end

hidden_to_class_learning_rates2 = [0.06, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15];

for lr = hidden_to_class_learning_rates2
  a4_main(300, .02, lr, 1000)
end
  