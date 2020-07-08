from keras.models import model_from_json, Model

import glob


json_save = './ae/weight_save/'
json_file = open(json_save+"model.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(json_save+"model.h5") 
import ipdb; ipdb.set_trace()

