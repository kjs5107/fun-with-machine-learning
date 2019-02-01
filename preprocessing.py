import numpy as np
from sklearn import preprocessing


input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])


def mean_removal():
    data_standardized = preprocessing.scale(input_data)
    print("\nMean = ", data_standardized.mean(axis=0))
    print("Std deviation = ", data_standardized.std(axis=0))


def scaling():
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_scaled = data_scaler.fit_transform(input_data)
    print("\nMin max scaled data = ", data_scaled)


def normalization():
    data_normalized = preprocessing.normalize(input_data, norm='l1')
    print("\nL1 normalized data = ", data_normalized)


def binarization():
    data_binarized = preprocessing.Binarizer(threshold=1.4).transform(input_data)
    print("\nBinarized data =", data_binarized)


def one_hot_encoding():
    encoder = preprocessing.OneHotEncoder(categories='auto')
    encoder.fit([  [0, 2, 1, 12],
               [1, 3, 5, 3],
               [2, 3, 2, 12],
               [1, 2, 4, 3]
    ])
    encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
    print("\nEncoded vector =", encoded_vector)


def main():
    mean_removal()
    scaling()
    normalization()
    binarization()
    one_hot_encoding()


if __name__ == "__main__":
    main()
