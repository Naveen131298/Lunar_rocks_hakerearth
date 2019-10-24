from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import pandas as pd
# load and prepare the image
CATEGORIES = ["Large", "Small"]
list=[]
df=pd.DataFrame()
df=pd.read_csv('test.csv')
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(72, 48))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 72, 48, 3)
	return img

# load an image and predict the class

def run_example():
	# load the image
	i=7534
	for i in range(7534):
		img = load_image('DataSet/Test Images/'+df['Image_File'][i])
		# load model
		model = tf.keras.models.load_model("64x3-CNN.model")
		# predict the class
		result = model.predict(img)
		list.append(CATEGORIES[int(result[0])])
		print(CATEGORIES[int(result[0])])
# entry point, run the example
run_example()
print(df)

