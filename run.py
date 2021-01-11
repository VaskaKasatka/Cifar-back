from web_api.app import create_app
from web_api.config import DevelopmentConfig, ProductionConfig


application = create_app(config_object=ProductionConfig)

if __name__ == '__main__':
    application.run(host='0.0.0.0')