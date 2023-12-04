% Function definition: y = x^2
f = @(x) x.^2;

% Adagrad Algorithm
learning_rate = 0.01;
max_iterations = 64;
initial_x = 10; % Starting point

x_values_adagrad = zeros(1, max_iterations);
y_values_adagrad = zeros(1, max_iterations);

cumulative_squared_gradients = 0; % Initialize cumulative squared gradients

% Optimization process with Adagrad
for i = 1:max_iterations
    gradient = 2 * initial_x; % Derivative of x^2 is 2x
    
    % Update cumulative squared gradients
    cumulative_squared_gradients = cumulative_squared_gradients + gradient^2;
    
    % Update weights using Adagrad
    initial_x = initial_x - (learning_rate / sqrt(cumulative_squared_gradients)) * gradient;
    
    % Save results
    x_values_adagrad(i) = initial_x;
    y_values_adagrad(i) = f(initial_x);
end

% Display the results
fprintf('Initial x: %.4f\n', x_values_adagrad(1));
fprintf('Optimized x: %.4f\n', x_values_adagrad(end));

% Plotting the graphs with Adagrad
figure;
subplot(2,1,1);
plot(x_range, y_range, 'LineWidth', 2);
hold on;
scatter(x_values_adagrad, y_values_adagrad, 'm', 'filled');
title('Adagrad: Finding the Minimum');
xlabel('x');
ylabel('y');
legend('y = x^2', 'Adagrad Steps', 'Location', 'Best');
grid on;
hold off;

% Plot the function values over iterations with Adagrad
subplot(2,1,2);
plot(1:max_iterations, y_values_adagrad, 'mo-', 'LineWidth', 2);
title('Function Values at Each Step with Adagrad');
xlabel('Iteration');
ylabel('y');
grid on;
