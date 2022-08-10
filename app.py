import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('kkn22.pkl', 'rb'))
model = pickle.load(open('nb_model22.pkl', 'rb'))
model = pickle.load(open('dt_model22.pkl', 'rb'))
model = pickle.load(open('rbf22_model.pkl', 'rb')) 


@app.route('/')
def home():
  
    return render_template("index2.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    CustomerID = float(request.args.get('CustomerID'))
    Surname = float(request.args.get('Surname'))
    CreditSocre = float(request.args.get('CreditSocre'))
    Geography = float(request.args.get('Geography'))
    Gender=float(request.args.get('Gender'))
    Age = float(request.args.get('Age'))
    Tenure = float(request.args.get('Tenure'))
    Blance = float(request.args.get('Blance'))
    NumofProducts = float(request.args.get('NumofProducts'))
    HasCrCrad = float(request.args.get('HasCrCrad'))
    IsActiveMember = float(request.args.get('IsActiveMember'))
    EstimatedSalary = float(request.args.get('EstimatedSalary'))
    
    prediction = model.predict([[ CustomerID, Surname, CreditSocre,Geography,Gender,Age,Tenure,Blance,NumofProducts,HasCrCrad,IsActiveMember,EstimatedSalary]])
    
        
    return render_template('index2.html', prediction_text='All Model  has predicted Exited for given Data is : {}'.format(prediction))
   
if __name__ == "__main__":
    app.run(debug=True)