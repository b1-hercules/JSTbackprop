import numpy as np
import pandas as pd
import tensorflow as tf
import json
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder ='templates',static_url_path='/static')

# Pandas read CSV
sf_train = pd.read_csv('p5_training_data.csv')

# korelasi matriks ke target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# buang kolom
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# baca CSV
sf_train = pd.read_csv('p5_training_data.csv')

# Korelasi matriks target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# buang kolom
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# panda baca data Val
sf_val = pd.read_csv('p5_val_data.csv')

# Buang kolom
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Get Pandas array value (Convert ke array NumPy)
train_data = sf_train.values
val_data = sf_val.values

# Use columns 2-akhir sbg Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 sebagai Output/Target (pake One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

# bikin jaringan
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# aktivasi softmax baut klasifikasi multiklas
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)

# Compile model dengan Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Training model dan pake validation data
model.fit(train_x, train_y, batch_size=10, epochs=5000, verbose=1, validation_data=(val_x, val_y))
model.save_weights('weights.h5')

# Predict semual Validation data
predict = model.predict(val_x)

# Visualisasi Prediksi
#df = pd.DataFrame(predict)
#df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
#df.index = val_data[:,0]

#web server pake flask buat render template
@app.route('/index')
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/dota2', methods=['GET'])
def dota():
	'''
	df = pd.DataFrame(predict)
	df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
	df.index = val_data[:,0]
	data = df.to_dict(orient='index')
	return jsonify({'data':data})
	'''
	df = pd.DataFrame(predict)
	df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
	df.index = val_data[:,0]
	temp = df.to_dict('records')
	columnNames = df.columns.values
	rowwNames = df.index
	return render_template('record.html', records=temp, colnames=columnNames,len=len(rowwNames), rownames=rowwNames)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
