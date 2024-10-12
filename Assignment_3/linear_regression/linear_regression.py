import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    data.columns = ['Study Hours', 'Pass']
    return data

def calculate_parameters(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    covariance = np.sum((x - mean_x) * (y - mean_y))

    variance = np.sum((x - mean_x) ** 2)

    slope = covariance / variance
    intercept = mean_y - slope * mean_x

    return slope, intercept


def plot_regression_line(x, y, slope, intercept):
    plt.scatter(x, y, color='blue', label='Data Points')  
    plt.plot(x, slope * x + intercept, color='red', label='Regression Line')  
    plt.xlabel('Study Hours')
    plt.ylabel('Pass')
    plt.title('Linear Regression on Study Hours vs Pass')
    plt.legend()
    plt.grid()
    plt.savefig('../results/linear_regression_plot.png') 
    plt.show()

def main():
    filename = 'linear_regression_data.csv'
    data = load_data(filename)

    print(data.head())  

    x = data['Study Hours'].values
    y = data['Pass'].values  

    slope, intercept = calculate_parameters(x, y)

    print(f'Slope: {slope}')
    print(f'Intercept: {intercept}')

    plot_regression_line(x, y, slope, intercept)


if __name__ == '__main__':
    main()
