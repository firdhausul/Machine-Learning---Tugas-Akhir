import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('knnbagg.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if(prediction)==0:
       return render_template('prediksi.html', prediction_text='" - Prediksi Barang tidak perlu Restock -"')
    else:
       return render_template('prediksi.html', prediction_text='" - Prediksi Barang perlu Restock - "')
    
 
    # return render_template('index.html', prediction_text='Prediksi Barang {}'.format(output))
    # return render_template('index.html', prediction_text='Prediksi Barang {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    prediction = model.predict_proba([np.array(list(data.values()))])
    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
