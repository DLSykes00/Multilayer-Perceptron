% Author: Daniel Sykes
% Date: Dec 2021
% MATLAB Version: 2021b

clear; clc;
load 'data/qsar.mat';

% Split up data 80:20
[training_data, testing_data] = split_data(D, 0.2, true);

% Create MLP
mlp_model = mlp(); % (Custom mlp implementation "mlp.m")
 
% Train MLP
mlp_model.train(training_data(:, 1:10), training_data(:, 11));

% Plot error^2 vs iterations
figure('Name', 'Loss');
plot(movmean(mlp_model.loss, mlp_model.iterations/50));
ylabel("Error ^{2}"); xlabel("Iterations");

% See how it performs with testing data
test_mlp(mlp_model, testing_data);

% Generates confusion matrix and accuracy for a given model & test data.
function accuracy = test_mlp(mlp_model, testing_data)
    for set = 1 : size(testing_data, 1)
        prediction = mlp_model.predict(testing_data(set, 1:10));
        model_pred(set) = prediction;
    end
    model_pred = model_pred';
    model_target = testing_data(:, 11);
    
    % Generate confusion matrix
    figure('Name', 'Confusion Matrix');
    confusion_matrix = confusionchart(model_pred, model_target);
    
    % Calculate accuracy
    correct_vals = sum((model_target-model_pred) == 0);
    accuracy = correct_vals / size(model_target, 1); 
    fprintf("Classification Accuracy: %0.3f \n", accuracy);
end

% Randomly split data into training and testing set
function [train, test] = split_data(data, test_split, deterministic)
    if(nargin > 2 && deterministic) rng(10000); end % rng gives a deterministic result
    data_length = size(data, 1);
    data = data(randperm(data_length), :); % Put data in random order
    test = data(1:test_split*data_length, :); 
    train = data(test_split*data_length:end, :);
end

% Test different values of hyperparameters
function [optimal] = optimise_paramter(model, train_data, test_data, parameter, p_min, p_max, delta)
    values = [];
    values_index = 1;
    for value = p_min:delta:p_max
        switch parameter
            case 'rate'
                model.rate = value;
            case 'neurons'
                model.no_neurons = value;
            case 'regularisation'
                model.regularisation = value;
            otherwise
                disp("Invalid parameter");
                return;
        end
        
        model.train(train_data(:, 1:10), train_data(:, 11));    
        
        values(values_index, 1) = value;
        values(values_index, 2) = test_mlp(model, test_data);
        values_index = values_index + 1;
    end
    [val, optimal_row] = max(values(:, 2));
    optimal = values(optimal_row, 1);
    figure;

    plot(values(:, 1), movmean(values(:, 2), 100))
end
