from sklearn.preprocessing import OneHotEncoder

# Converting categorical variables into a form which is better understood by the machine learning algorithms.
def one_hot_encoded(y):
    values = y
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = values.reshape(len(values), 1)
    one_hot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded