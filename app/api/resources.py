import warnings
warnings.filterwarnings("ignore")

from flask_restful import Resource, reqparse

from . import api
from .decorators import post_response
from src.config import model_id, THRESHOLD, SPAM_LABEL, HAM_LABEL, ROUND
from src.helpers import load_model, prepare_sms

model = load_model(name=model_id)
parser = reqparse.RequestParser()
parser.add_argument('text', help='This field cannot be blank', required=True)

class SpamDetector(Resource):

    @post_response
    def post(self):
        post_data = parser.parse_args()
        text = post_data.get('text')
        input_data = prepare_sms(text)
        ham_proba, spam_proba = model.predict_proba(input_data)[0]

        if ham_proba >= THRESHOLD:
            label = HAM_LABEL
            confidence = ham_proba
        else:
            label = SPAM_LABEL
            confidence = spam_proba

        responseObject = {
            'status': 'success',
            'label': label,
            'confidence': round(confidence, ROUND)}

        return responseObject, 200
