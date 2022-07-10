from flask import Flask
from model import *

app = Flask(__name__)

@app.route('/')
def index():
    return os.getcwd()

@app.route('/<client_id>')
def predict(client_id):

    client_df = get_client_row(client_id)
    prediction, probability = predict_client(client_id, client_df)

    return {
        'client': client_df.iloc[0].to_dict(),
        'prediction': prediction,
        'probability': probability
    }

@app.route('/features')
def feature_importances():
    model = get_model()
    features = pd.DataFrame({
        'name': model.feature_name_,
        'importance': model.feature_importances_
    })

    return dict(zip(features.name, features.importance))