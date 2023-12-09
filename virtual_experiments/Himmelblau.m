% Function definition: Himmelblau's function
himmelblau = @(x, y) (x.^2 + y - 11).^2 + (x + y.^2 - 7).^2;

% Parameters
max_iterations = 512;
initial_x = [7; 7]; % Initial point [x, y]
epsilon = 1e-8;
beta_1 = 0.9;
beta_2 = 0.999;

% Initialization
x_values = zeros(5, max_iterations, 2);
y_values = zeros(5, max_iterations);

% Optimization process
optimizers = {'SGD', 'SGDM', 'RMSProp', 'Adagrad', 'Adam'};
learning_rates = [0.001, 0.001, 0.01, 0.01, 0.01]; % Specify the learning rates

% Create a single plot
figure;

% Plot Himmelblau's function
[x_himmelblau, y_himmelblau] = meshgrid(-6:0.1:6, -6:0.1:6);
z_himmelblau = himmelblau(x_himmelblau, y_himmelblau);
subplot(2, 1, 1);
mesh(x_himmelblau, y_himmelblau, z_himmelblau);
title("Himmelblau's Function");

% Plot optimization results for Himmelblau's function
subplot(2, 1, 2);
for opt_idx = 1:5
    current_optimizer = optimizers{opt_idx};
    fprintf('Running optimization with %s...\n', current_optimizer);

    % Initialize variables based on optimizer
    switch current_optimizer
        case 'SGD'
            x = initial_x;
            learning_rate = learning_rates(opt_idx);
        case 'SGDM'
            x = initial_x;
            learning_rate = learning_rates(opt_idx);
            m = 0;
        case 'Adagrad'
            x = initial_x;
            v = 0;
            learning_rate = learning_rates(opt_idx);
        case 'RMSProp'
            x = initial_x;
            v = 0;
            learning_rate = learning_rates(opt_idx);
        case 'Adam'
            x = initial_x;
            m = 0;
            v = 0;
            t = 0;
            learning_rate = learning_rates(opt_idx);
    end

    % Optimization loop
    for i = 1:max_iterations
        gradient = [4 * x(1) * (x(1)^2 + x(2) - 11) + 2 * (x(1) + x(2)^2 - 7);
                    2 * (x(1)^2 + x(2) - 11) + 4 * x(2) * (x(1) + x(2)^2 - 7)];

        % Update weights based on optimizer
        switch current_optimizer
            case 'SGD'
                x = x - learning_rate * gradient;
            case 'SGDM'
                m = beta_1 * m + (1 - beta_1) * gradient;
                x = x - learning_rate * m;
            case 'Adagrad'
                v = v + gradient.^2;
                x = x - (learning_rate / (sqrt(v) + epsilon)) * gradient;
            case 'RMSProp'
                t = t + 1;
                v = beta_2 * v + (1 - beta_2) * gradient.^2;
                v_hat = v / (1 - beta_2^t);
                x = x - (learning_rate / (sqrt(v_hat) + epsilon)) * gradient;
            case 'Adam'
                t = t + 1;
                m = beta_1 * m + (1 - beta_1) * gradient;
                v = beta_2 * v + (1 - beta_2) * gradient.^2;
                m_hat = m / (1 - beta_1^t);
                v_hat = v / (1 - beta_2^t);
                x = x - (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat;
        end

        % Save results
        x_values(opt_idx, i, :) = x;
        y_values(opt_idx, i) = himmelblau(x(1), x(2));
    end

    % Plot results for the current optimizer on the same graph
    plot(1:max_iterations, y_values(opt_idx, :), 'LineWidth', 2, 'DisplayName', current_optimizer);
    hold on;
end

% Finalize the plot
title("Optimizers on Himmelblau's Function");
xlabel('Iteration');
ylabel('f(x, y)');
legend('Location', 'Best');
grid on;
hold off;
