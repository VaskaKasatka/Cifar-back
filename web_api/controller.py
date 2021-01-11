from flask import Blueprint, request, jsonify
from cnn_model.predict import make_prediction
from cnn_model import __version__ as model_version

from web_api import __version__ as api_version
from web_api.config import get_logger

import json


_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status ok')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        img_url = json_data.get("img_url")
        img_id = json_data.get("img_id")

        result = make_prediction(img_url=img_url, img_id=img_id)
        _logger.info(f'Outputs: {result}')

        prediction = result.get('prediction')
        score = float(result.get('score'))
        version = result.get('version')

        # TODO: обработка ошибок
        errors = None

        return jsonify({'prediction': prediction,
                        'score': score,
                        'version': version,
                        'errors': errors})
