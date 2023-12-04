% Function definition: y = x^2
f = @(x) x.^2;

% Gradient Descent Algorithm
learning_rate = 0.001;
max_iterations = 64;
initial_x = 10; % Starting point

x_values = zeros(1, max_iterations);
y_values = zeros(1, max_iterations);

% Optimization process
for i = 1:max_iterations
    gradient = 2 * initial_x; % Derivative of x^2 is 2x
    initial_x = initial_x - learning_rate * gradient;
    
    % Save results
    x_values(i) = initial_x;
    y_values(i) = f(initial_x);
end

% Display the results
fprintf('Initial x: %.4f\n', x_values(1));
fprintf('Optimized x: %.4f\n', x_values(end));

% Plotting the graphs
x_range = -13:0.1:13;
y_range = f(x_range);

figure;
subplot(2,1,1);
plot(x_range, y_range, 'LineWidth', 2);
hold on;
scatter(x_values, y_values, 'r', 'filled');
title('Gradient Descent');
xlabel('x');
ylabel('y');
legend('y = x^2', 'Gradient Descent Steps', 'Location', 'Best');
grid on;
hold off;

% Plot the function values over iterations
subplot(2,1,2);
plot(1:max_iterations, y_values, 'bo-', 'LineWidth', 2);
title('Gradient Descent');
xlabel('Iteration');
ylabel('y');
grid on;
