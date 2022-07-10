import pandas as pd
import pickle

TEST_SET_PATH = '../../dashboard/data/test_set.csv'
MODEL_PATH = '../model/lgbm.pkl'
PREDICTION_THRESHOLD = 0.05

def get_model():
    return pickle.load(open(MODEL_PATH, 'rb'))


def get_client_row(client_id):
    client_id = int(client_id)

    # Loading the test dataset
    test_df = pd.read_csv(TEST_SET_PATH, index_col=0)
    client_df = test_df[test_df['SK_ID_CURR'] == client_id]

    return client_df


def predict_client(client_id, client_df=None):
    if client_df is None:
        client_df = get_client_row(client_id)

    if len(client_df) == 1:
        model = get_model()
        predictions = model.predict_proba(client_df.drop(columns=['SK_ID_CURR']))

        if predictions[0][0] > PREDICTION_THRESHOLD:
            return 0, predictions[0][0]
        else:
            return 1, predictions[0][1]

    return None, None