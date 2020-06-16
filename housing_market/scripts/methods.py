import pandas as pd

def write_submissions(file_name, test_df, predictions):
    test_df.Id = test_df.Id.astype('int32')
    output = pd.DataFrame({
        'Id': test_df.Id, 'SalePrice': predictions
    })
    output.to_csv(file_name, index=False)


def get_categorical_columns(data_df):
    return list(data_df.select_dtypes(include=['category', 'object']))


def get_numeric_columns(data_df):
    return list(data_df.select_dtypes(exclude=['category', 'object']))


def read_train_test_data():
    train_df = pd.read_csv('../home-data-for-ml-course/train.csv', index_col='Id')
    test_df = pd.read_csv('../home-data-for-ml-course/test.csv', index_col='Id')

    print("Shape of Train Data: " + str(train_df.shape))
    print("Shape of Test Data: " + str(test_df.shape))

    return train_df, test_df


# function to return the global name of an object
def name_of_global_obj(xx):
    return [objname for objname, oid in globals().items()
            if id(oid)==id(xx)][0]

# function to return key for any value
def get_key(dict, val):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

