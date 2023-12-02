import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=1.0.bin'

'''
with open() is concerned with opening and closing the file, but the pickle.load()
would put into the variables dv and model necessary stuff such that they
can be used throught this script! 

So it will open the file, pickle.load() would put the stuff in those 
dv and model variables and then they could be used anywhere!
Finally, the file would be closed such that it doesn't use memory!


'''

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in) 

app = Flask('churn')

#POST is used to send data to a server to create/update a resource.
@app.route('/predict', methods=['POST'])
def predict():
    #Assuming that we are sending data in the JSON type, and 
    #that dv.transform() needs dict type this will turn that
    #request into dict:
    '''
    request.get_json() will return a Python dictionary with keys 
    and values corresponding to the JSON fields.
    '''
    customer = request.get_json() 

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    #Marketing team for sending promotional emails doesn't understand
    #for which kind of probability it will send a promotional email
    #so we will be explicit here and send True if p>=0.5
    churn = y_pred >= 0.5

    #We have to use float() and bool() since JSON doesn't understand
    #numpy's version of float and bool, only Pythons.
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result) #Turn the response to JSON type!


#Gunicorn doesn't care about this line because it is looking for a module
#and in particullar is interested in app = Flask('churn') part.
#With --bind 0.0.0.0:9696 we are opening port on that server. 
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
