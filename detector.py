import tensorflow as tf
from dataManager import DataManager

class Detector:

  def __init__(self, model_path):
    try:
      self.model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
      print("Error: Model not found")
      self.model = None
      
    self.data = DataManager()

  def detect(self, img):
    if not self.model:
      return None
    
    try:
      img = self.data.preprocess(img)
    except:
      print("Error preprocessing image")
      return None
    
    prediction = self.model.predict(img[None, :])
    print(f"Prediction: {prediction}")
    
    return prediction

detector = Detector('model.h5')
predictions = detector.detect('test.jpg')