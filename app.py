from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('possum.csv')
#lets just look at how age, hdlengh, skullw, taill and footlgth affect totlength
possum_data = data[['age','hdlngth','skullw','totlngth','taill', 'footlgth']]
#removing any rows containing missing data
possum_data = possum_data.dropna()

X = possum_data[['age','hdlngth','skullw','taill', 'footlgth']]#factors, e.g. number of bedrooms in a house, area income
y= possum_data[['totlngth']] #target column, e.g. house price

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =101)
lm = LinearRegression()  
lm.fit(X_train, y_train) #train the model

pickle.dump(lm, open('possum_length.pickle', 'wb'))
# 'wb' means we're storing it as bytes

app = Flask(__name__)
model = pickle.load(open('possum_length.pickle', 'rb'))

@app.route('/', methods = ['GET', 'POST'], endpoint = 'home')  #set the route of the request, / means return to home page
def home():
    return render_template('index.html')

@app.route('/Predict/', endpoint = 'length_predict', methods = ['GET', 'POST'])
def length_predict():
    
    age = request.form.get("age")
    hdlngth = request.form.get("hdlngth")
    skullw = request.form.get("skullw")
    taill = request.form.get("taill")
    footlgth = request.form.get("footlgth")  

    test_df = pd.DataFrame({'age':[age], 'hdlngth':[hdlngth], 'skullw':[skullw], 'taill':[taill],'footlgth':[footlgth]})
    test_df = test_df.dropna()
  
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    pred_total_length = model.predict(test_df)
    
    output = pred_total_length
    # return str(footlgth)
    return render_template('index.html', prediction_text = 'Total possum length should be {}'.format(output))

if __name__ == '__main__':
    app.run(debug = True, use_reloader = False) #run the app, can take app.run(port=) argument