% Function definition: y = x^2
f = @(x) x.^2;

% Parameters
learning_rate = 0.01;
max_iterations = 64;
initial_x = 7; 
epsilon = 1e-8;
beta_1 = 0.9;
beta_2 = 0.999;

% Initialization
x_values = zeros(5, max_iterations);
y_values = zeros(5, max_iterations);

% Optimization process
optimizers = {'Gradient Descent', 'Momentum', 'RMSProp', 'Adagrad', 'Adam'};

% Create a single plot
figure;

for opt_idx = 1:5
    current_optimizer = optimizers{opt_idx};
    fprintf('Running optimization with %s...\n', current_optimizer);
    
    % Initialize variables based on optimizer
    switch current_optimizer
        case 'Gradient Descent'
            x = initial_x;
        case 'Momentum'
            x = initial_x;
            momentum = 0;
        case 'RMSProp'
            x = initial_x;
            cumulative_squared_gradients = 0;
        case 'Adagrad'
            x = initial_x;
            cumulative_squared_gradients = 0;
        case 'Adam'
            x = initial_x;
            m = 0;
            v = 0;
            t = 0;
    end
    
    % Optimization loop
    for i = 1:max_iterations
        gradient = 2 * x; % Derivative of x^2 is 2x
        
        % Update weights based on optimizer
        switch current_optimizer
            case 'Gradient Descent'
                x = x - learning_rate * gradient;
            case 'Momentum'
                momentum = beta_1 * momentum + (1 - beta_1) * gradient;
                x = x - learning_rate * momentum;
            case 'RMSProp'
                cumulative_squared_gradients = beta_2 * cumulative_squared_gradients + (1 - beta_2) * gradient^2;
                x = x - (learning_rate / (sqrt(cumulative_squared_gradients) + epsilon)) * gradient;
            case 'Adagrad'
                cumulative_squared_gradients = cumulative_squared_gradients + gradient^2;
                x = x - (learning_rate / (sqrt(cumulative_squared_gradients) + epsilon)) * gradient;
            case 'Adam'
                t = t + 1;
                m = beta_1 * m + (1 - beta_1) * gradient;
                v = beta_2 * v + (1 - beta_2) * gradient^2;
                m_hat = m / (1 - beta_1^t);
                v_hat = v / (1 - beta_2^t);
                x = x - (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat;
        end
        
        % Save results
        x_values(opt_idx, i) = x;
        y_values(opt_idx, i) = f(x);
    end
    
    % Plot results for current optimizer on the same graph
    plot(1:max_iterations, y_values(opt_idx, :), 'LineWidth', 2, 'DisplayName', current_optimizer);
    hold on;
end

% Finalize the plot
title('Optimizers on y = x^2');
xlabel('Iteration');
ylabel('y');
legend('Location', 'Best');
grid on;
hold off;
