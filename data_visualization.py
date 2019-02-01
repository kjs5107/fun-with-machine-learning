import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# read in the csv and setup the dataset
data = './assets/iris_df.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(data, names=names)


def univariate_plots():
    # box & whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

    # histogram plots
    dataset.hist()


def multivariate_plots():
    scatter_matrix(dataset)


def main():
    univariate_plots()
    multivariate_plots()
    # display all plots
    plt.show()


if __name__ == "__main__":
    main()
