import numpy as np
import pickle
import logging
#from Modular_prog1 import Pipeline
#Pipe = Pipeline()
from flask import Flask,render_template,request

application=Flask(__name__)
app=application
Pipe=pickle.load(open('pipe.pkl','rb'))
logging.basicConfig(filename='Prediction_pipeline.log',level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')


class Prediction_pipeline:
    def prediction(self,Pipe,input_data):
        try:
            self.ip=list(input_data.values())
            self.Pipe=Pipe
            #self.ip=input_data
            self.test_data=np.array(self.ip).reshape(1,12)
            self.result=self.Pipe.predict(self.test_data)
            
            logging.info("Prediction pipeline class from prediction pipeline executed successfully")
            return self.result[0]
        except Exception as e:
            logging.error("Exception occured at Prediction pipeline class from prediction pipeline")
            logging.error(e)
            


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
        # Get input data from form
            input_data = {
                'Age': int(request.form['Age']),
                'Workclass': request.form['Workclass'],
                'Fnlwgt': int(request.form['Fnlwgt']),
                'Education-num': int(request.form['Education-num']),
                'Occupation': request.form['Occupation'],
                'Relationship': request.form['Relationship'],
                'Race': request.form['Race'],
                'Sex': request.form['Sex'],
                'Capital-gain': int(request.form['Capital-gain']),
                'Capital-loss': int(request.form['Capital-loss']),
                'Hours-per-week': int(request.form['Hours-per-week']),
                'Native-country': request.form['Native-country']
            }
            # Predict using input data
            Prediction=Prediction_pipeline()
            predicted_value = Prediction.prediction(Pipe,input_data)
            logging.info('Home page executed successfully in POST method')
            return render_template('result.html', predicted_value=predicted_value)
        else:
            logging.info('Home page executed successfully in GET method')
            return render_template('form.html')
    except Exception as e:
        logging.error('Exception occurred in home page')
        logging.error(e)

if __name__ == '__main__':
    app.run(debug=True)







