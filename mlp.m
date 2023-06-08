% Author: Daniel Sykes
% Date: Dec 2021
% MATLAB Version: 2021b

% Basic MLP implementation
classdef mlp < handle
    properties (Access = public) % PUBLIC PROPERTIES
        no_neurons {mustBeInteger, mustBePositive, mustBeNonzero} = 100;
        rate {mustBeNumeric, mustBePositive, mustBeNonzero} = 1e-3;
        iterations {mustBeInteger, mustBePositive, mustBeNonzero} = 5e4;
        regularisation {mustBeNumeric, mustBeNonnegative} = 1e-3;

        input_weights = [];
        hidden_weights = []; 
        loss = [];
    end
    
    properties (Access = private)   
    end

    methods (Access = public) % PUBLIC METHODS
        function obj = mlp() % Constructor
        end

        function obj = train(obj, input_data, values) 
            N = size(input_data, 2);
            input_sets = size(input_data, 1);
            
            % Append bias term (1's) to input_data
            input_data = [input_data ones(input_sets, 1)];

            % Initialise weights uniformally -1 to 1
            obj.input_weights = [(2 * rand([obj.no_neurons, N]) - 1) ones(obj.no_neurons,1)];     
            obj.input_weights = [obj.input_weights; zeros(1, N) inf(1)]; % Inf to keep bias 1 for next layer (in case of sigmoid) as sig(inf) = 1  
            obj.hidden_weights = [(2 * rand([1, obj.no_neurons]) - 1)  1];
            
            shuffled_indices = randperm(input_sets);
            shuffled_index = 1;

            % MLP ANN Algorithm 
            for pass = 1 : obj.iterations
                if (mod(pass, 1000) == 0) 
                    clc;
                    fprintf("Passes Completed: %d\n", pass); 
                end
                if (mod(pass, input_sets) == 0)
                    shuffled_indices = randperm(input_sets);
                    shuffled_index = 1;
                end
                data_set = shuffled_indices(shuffled_index);
                input_neurons = input_data(data_set, :)';
                targets = values(data_set, :)';
                shuffled_index = shuffled_index + 1;

                % Forward pass
                hidden_neurons = obj.sigmoid(obj.input_weights * input_neurons);
                output_neurons = obj.sigmoid(obj.hidden_weights * hidden_neurons);
                
                % Backward pass
                errors_output = targets - output_neurons; % Errors
                dWh = (errors_output .* (output_neurons .* (1 - output_neurons))) * hidden_neurons'; % Delta weights
                dWh = dWh - [(obj.regularisation .* obj.hidden_weights .^ 2)]; % Regularisation
                obj.hidden_weights = obj.hidden_weights + (obj.rate .* dWh); % Change weights by deltas

                errors_hidden = (obj.hidden_weights') * errors_output; % Errors
                dWx = (errors_hidden .* (hidden_neurons .* (1 - hidden_neurons))) * input_neurons'; % Delta weights
                obj.input_weights = obj.input_weights + (obj.rate .* dWx); % Change weights by deltas

                obj.loss(pass) = errors_output^2;
            end
            
            fprintf("Model Trained\n");
        end

        function prediction = predict(obj, input_data)
            input_dimensions = size(input_data, 2);
            input_sets = size(input_data, 1);

            % Append bias term row (1's) to input_data
            input_data = [input_data ones(input_sets, 1)];           
            input_neurons = input_data';

            % Forward pass
            hidden_neurons = obj.sigmoid(obj.input_weights * input_neurons);
            output_neurons = obj.sigmoid(obj.hidden_weights * hidden_neurons);
            
            if (output_neurons >= 0.5) 
                prediction = 1;
            else 
                prediction = 0; 
            end
        end
    end

    methods (Access = private) % PRIVATE METHODS
        function result = sigmoid(obj, x)
            result = 1 ./ (1 + exp(-x));
        end
    end
end
