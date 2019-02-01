import pandas


def main():
    # set up the dataset
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
             'Age', 'Class']
    dataset = pandas.read_csv("./assets/pima-indians-diabetes.csv", names=names)

    # try out some summarization
    print("\n================================Basic Summary===================================\n")
    print("The data is this long:", dataset.shape)
    print("First 20 rows:\n", dataset.head(20))
    print("\n==========================Some Statistical Manipulations=============================\n")
    print(dataset.describe())
    print(dataset.groupby('Pregnancies').size())


if __name__ == "__main__":
    main()



