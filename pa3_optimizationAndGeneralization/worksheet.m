% Let's do an initial run with with learning rate 0.005 and no momentum: run a3(0, 10, 70, 0.005, 0, false, 4)
a3(0, 10, 70, 0.005, 0, false, 4)


% Let's try a variety of learning rates, to find out which works best. 
%   We'll try 0.002, 0.01, 0.05, 0.2, 1.0, 5.0, and 20.0. 
%   We'll try all of those both without momentum (i.e. momentum=0.0 in the program) and with momentum (i.e. momentum=0.9 in the program),
%     so we have a total of 7 x 2 = 14 experiments to run.
learning_rates = [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0]
momentums = [0.0, 0.9]
for(mom = momentums)
  for(lr = learning_rates)
  
    fprintf("\n---------------------------------------------------------------------------------");
    fprintf("\nLEARNING with %d learning rate and %d momentum\n", lr, mom);
    a3(0, 10, 70, lr, mom, false, 4)
  end
end


% Now we're interested mostly in the classification loss on the validation data.
% What is the validation data classification loss now? ( a3(0, 200, 1000, 0.35, 0.9, false, 100) )
a3(0, 200, 1000, 0.35, 0.9, false, 100)


% Run the experiment with the early stopping parameter set to true.
a3(0, 200, 1000, 0.35, 0.9, true, 100)


% Let's turn off early stopping, and instead investigate weight decay
% Be careful to focus on the classification loss (i.e. without the weight decay loss)
weithg_decays = [0, 0.0001, 0.001, 0.1, 1, 10]
for(wd = weithg_decays)

  fprintf("\n---------------------------------------------------------------------------------");
  fprintf("\nLEARNING with %d weigth decay\n", wd);
  a3(wd, 200, 1000, 0.35, 0.9, false, 100)
end


% Yet another regularization strategy is reducing the number of model parameters
% Turn off the weight decay, and instead try the following hidden layer sizes. 
hidden_nums = [200, 130, 100, 10, 30];
for(hn = hidden_nums)

  fprintf("\n---------------------------------------------------------------------------------");
  fprintf("\nLEARNING with %d hidden layers\n", hn);
  a3(0, hn, 1000, 0.35, 0.9, false, 100)
end


% Which number of hidden units works best that way, i.e. with early stopping?
hidden_nums = [236, 113, 83, 37, 18];
for(hn = hidden_nums)

  fprintf("\n---------------------------------------------------------------------------------");
  fprintf("\nLEARNING with %d hidden layers\n", hn);
  a3(0, hn, 1000, 0.35, 0.9, true, 100)
end


% For the settings that you chose on the previous question, what is the test data classification error rate?
a3(0, 37, 1000, 0.35, 0.9, true, 100)

