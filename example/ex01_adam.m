% Function definition: y = x^2
f = @(x) x.^2;

% Adam Algorithm
learning_rate = 0.01;
beta_1 = 0.9;
beta_2 = 0.999;
epsilon = 1e-8;
max_iterations = 64;
initial_x = 10; % Starting point

x_values_adam_custom = zeros(1, max_iterations);
y_values_adam_custom = zeros(1, max_iterations);

m = 0; % Initialize first moment
v = 0; % Initialize second moment
t = 0; % Initialize time step

% Optimization process with Adam (custom)
for k = 1:max_iterations
    gradient = 2 * initial_x; % Derivative of x^2 is 2x
    
    % Update time step
    t = t + 1;
    
    % Update moments
    m = beta_1 * m + (1 - beta_1) * gradient;
    v = beta_2 * v + (1 - beta_2) * gradient^2;
    
    % Bias correction for moments
    m_hat = m / (1 - beta_1^t);
    v_hat = v / (1 - beta_2^t);
    
    % Update weights using Adam
    initial_x = initial_x - (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat;
    
    % Save results
    x_values_adam_custom(k) = initial_x;
    y_values_adam_custom(k) = f(initial_x);
end

% Display the results
fprintf('Initial x: %.4f\n', x_values_adam_custom(1));
fprintf('Optimized x: %.4f\n', x_values_adam_custom(end));

% Plotting the graphs with Adam (custom)
figure;
subplot(2,1,1);
plot(x_range, y_range, 'LineWidth', 2);
hold on;
scatter(x_values_adam_custom, y_values_adam_custom, 'c', 'filled');
title('Adam (Custom): Finding the Minimum');
xlabel('x');
ylabel('y');
legend('y = x^2', 'Adam Steps (Custom)', 'Location', 'Best');
grid on;
hold off;

% Plot the function values over iterations with Adam (custom)
subplot(2,1,2);
plot(1:max_iterations, y_values_adam_custom, 'co-', 'LineWidth', 2);
title('Function Values at Each Step with Adam (Custom)');
xlabel('Iteration');
ylabel('y');
grid on;
