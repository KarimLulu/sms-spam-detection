from flask import Flask
from flask_bootstrap import Bootstrap

bootstrap = Bootstrap()

def create_app():
    app = Flask(__name__)
    bootstrap.init_app(app)
    app.config.update(dict(SECRET_KEY="powerful secretkey",
                           WTF_CSRF_SECRET_KEY="a csrf secret key"))

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
