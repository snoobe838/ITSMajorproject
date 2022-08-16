from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)

    
model_RF=pickle.load(open('Major_RF.pkl', 'rb')) 
model_KNN=pickle.load(open('Major_KNN.pkl', 'rb')) 
model_K_SVM=pickle.load(open('Major_SVM_linear.pkl', 'rb')) 
model_DT=pickle.load(open('Major_DT.pkl', 'rb')) 
model_NB=pickle.load(open('Major_NB.pkl', 'rb')) 



@app.route('/')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
  
@app.route('/predict',methods=['GET'])

def predict():
    
     
    age = float(request.args.get('age'))
    gender = float(request.args.get('gender'))
    bmi = float(request.args.get('bmi'))
    phq = float(request.args.get('phq'))


 
    Model = (request.args.get('Model'))

    if Model=="Random Forest Classifier":
      prediction = model_RF.predict([[age, gender, bmi, phq]])

    elif Model=="Decision Tree Classifier":
      prediction = model_DT.predict([[age, gender, bmi, phq]])

    elif Model=="KNN Classifier":
      prediction = model_KNN.predict([[age, gender, bmi, phq]])

    elif Model=="SVM Classifier":
      prediction = model_K_SVM.predict([[age, gender, bmi, phq]])

    else:
      prediction = model_NB.predict([[age, gender, bmi, phq]])

    
    if prediction == [0]:
      return render_template('index.html', prediction_text="The Patient's condition is not well he is being Suicidal.", extra_text =" -- Prediction by " + Model)

    else :
      return render_template('index.html', prediction_text='The Patient is Normal.', extra_text =" -- Prediction by " + Model)

if __name__ == "__main__":
    app.run(debug=True)
