from flask import Flask

def create_app():

    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    with app.app_context():
        from . import routes
        app.register_blueprint(routes.main_bp)
        app.register_blueprint(routes.admin_bp)

    @app.after_request
    def add_header(response):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return app