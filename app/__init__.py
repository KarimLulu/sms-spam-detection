from flask import Flask

def create_app():
    app = Flask(__name__)

    from .api import api
    api.init_app(app)

    return app
