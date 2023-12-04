% Function definition: y = x^2
f = @(x) x.^2;

% Momentum Algorithm
learning_rate = 0.001;
beta = 0.9;
max_iterations = 64;
initial_x = 10; % Starting point

x_values_momentum = zeros(1, max_iterations);
y_values_momentum = zeros(1, max_iterations);

momentum = 0; % Initialize momentum

% Optimization process with Momentum
for i = 1:max_iterations
    gradient = 2 * initial_x; % Derivative of x^2 is 2x
    
    % Momentum update
    momentum = beta * momentum + (1 - beta) * gradient;
    
    % Update weights using the momentum and learning rate
    initial_x = initial_x - learning_rate * momentum;
    
    % Save results
    x_values_momentum(i) = initial_x;
    y_values_momentum(i) = f(initial_x);
end

% Display the results
fprintf('Initial x: %.4f\n', x_values_momentum(1));
fprintf('Optimized x: %.4f\n', x_values_momentum(end));

% Plotting the graphs with Momentum
figure;
subplot(2,1,1);
plot(x_range, y_range, 'LineWidth', 2);
hold on;
scatter(x_values_momentum, y_values_momentum, 'g', 'filled');
title('Gradient Descent with Momentum');
xlabel('x');
ylabel('y');
legend('y = x^2', 'Gradient Descent Steps with Momentum', 'Location', 'Best');
grid on;
hold off;

% Plot the function values over iterations with Momentum
subplot(2,1,2);
plot(1:max_iterations, y_values_momentum, 'go-', 'LineWidth', 2);
title('Gradient Descent with Momentum');
xlabel('Iteration');
ylabel('y');
grid on;
