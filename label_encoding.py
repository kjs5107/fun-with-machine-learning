from sklearn import preprocessing


def main():
    # Trying out encoding
    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['suzuki', 'ford', 'suzuki', 'toyota', 'ford', 'bmw']
    label_encoder.fit(input_classes)
    print("Class mapping:")
    for i, item in enumerate(label_encoder.classes_):
        print(item, '-->', i)

    labels = ['toyota', 'ford', 'suzuki']
    encoded_labels = label_encoder.transform(labels)
    print("\nLabels =", labels)
    print("Encoded labels =", list(encoded_labels))

    # Trying out Decoding
    encoded_labels = [3, 2, 0, 2, 1]
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    print("\nEncoded labels =", encoded_labels)
    print("Decoded labels =", list(decoded_labels))


if __name__ == "__main__":
    main()
