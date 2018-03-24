import pandas as pd
from numpy import *


def compute_error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b, m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / n) * (y - (m * x + b))
        m_gradient += -(2 / n) * x * (y - (m * x + b))
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, init_b, init_m, learning_rate, iterations):
    b = init_b
    m = init_m

    for i in range(iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    # Read dataset
    points = pd.read_csv(
        "/home/zuhaib/Code/Python/ML Tutorials/datasets/linear_regression/tut_linear_regression.csv")
    # Set hyperparameter
    learning_rate = 0.0001
    # Initial values for model (y = mx + b)
    init_b = 0
    init_m = 0
    iterations = 1000
    [b, m] = gradient_descent_runner(points, init_b, init_m, learning_rate, iterations)
    print(b)
    print(m)


if __name__ == '__main__':
    run()
