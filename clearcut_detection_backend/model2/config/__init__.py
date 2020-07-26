import yaml

with open('/model/predict_config.yml', 'r') as config:
    cfg = yaml.load(config, Loader=yaml.SafeLoader)

models = cfg['models']
save_path = cfg['prediction']['save_path']
threshold = cfg['prediction']['threshold']
input_size = cfg['prediction']['input_size']
db_string = cfg['database']['database_uri']
