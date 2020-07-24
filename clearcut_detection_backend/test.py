from test.predict import model_predict
from test.evaluation import model_evaluate

results, test_tile_path = model_predict()
model_evaluate(results, test_tile_path)
