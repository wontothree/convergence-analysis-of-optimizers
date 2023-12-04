% Function definition: y = x^2
f = @(x) x.^2;

% RMSProp Algorithm
learning_rate = 0.01;
beta_2 = 0.9;
epsilon = 1e-8;
max_iterations = 64;
initial_x = 10; % Starting point

x_values_rmsprop_custom = zeros(1, max_iterations);
y_values_rmsprop_custom = zeros(1, max_iterations);

v = 0; % Initialize second moment

% Optimization process with RMSProp
for k = 1:max_iterations
    gradient = 2 * initial_x; % Derivative of x^2 is 2x
    
    % Update moment
    v = beta_2 * v + (1 - beta_2) * gradient^2;
    
    % Update weights using RMSProp
    initial_x = initial_x - (learning_rate / (sqrt(v) + epsilon)) * gradient;
    
    % Save results
    x_values_rmsprop_custom(k) = initial_x;
    y_values_rmsprop_custom(k) = f(initial_x);
end

% Display the results
fprintf('Initial x: %.4f\n', x_values_rmsprop_custom(1));
fprintf('Optimized x: %.4f\n', x_values_rmsprop_custom(end));

% Plotting the graphs with RMSProp (custom)
figure;
subplot(2,1,1);
plot(x_range, y_range, 'LineWidth', 2);
hold on;
scatter(x_values_rmsprop_custom, y_values_rmsprop_custom, 'b', 'filled');
title('RMSProp (Custom): Finding the Minimum');
xlabel('x');
ylabel('y');
legend('y = x^2', 'RMSProp Steps (Custom)', 'Location', 'Best');
grid on;
hold off;

% Plot the function values over iterations with RMSProp (custom)
subplot(2,1,2);
plot(1:max_iterations, y_values_rmsprop_custom, 'bo-', 'LineWidth', 2);
title('Function Values at Each Step with RMSProp (Custom)');
xlabel('Iteration');
ylabel('y');
grid on;
