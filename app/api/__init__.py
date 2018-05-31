from flask_restful import Api

api = Api()

from .resources import SpamDetector

api.add_resource(SpamDetector, '/detector')
