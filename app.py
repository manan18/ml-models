from flask import Flask
from models import purchase_model
from models import top_performer
from flask import request, jsonify
from models import sales
import pickle

app = Flask(__name__)



@app.route('/predict-vendor', methods=['POST'])
def predict():
    data = request.get_json()
    result = purchase_model.predict(data)
    return jsonify(result.to_dict(orient='records'))
#     #new_data = {
#     'VendorID': ['Vendor101', 'Vendor102', 'Vendor103', 'Vendor104'],
#     "ProductID": ["Product1", "Product1", "Product2" , "Product2"],
#     "Price": [110, 105 , 210, 190],
#     "DeliveryDays": [4, 3, 7, 14],
#     "QualityRating": [5, 4, 3, 5],
#     "FulfillmentRate": [98, 97, 95, 96],
#     "OnTimeDeliveryRate": [92, 95, 90, 96],
#     "OrderAccuracyRate": [94, 96, 100, 94],
#     "FinancialHealthScore": [0.88, 0.90, 0.92, 0.95],
#     "ResponseTime": [20, 18, 12, 20]
# }data ka format


@app.route('/predict-employee', methods=['POST'])
def predict1():

    data = request.get_json()
    # print(type(data))
    result = top_performer.predict1(data)
    # print(result)
    return jsonify(result)
# # X_new_data = {
# #     'Percentage': [0.75, 0.60, 0.80],  # Example values for Percentage feature
# #     'Duration': [5, 7, 4],              # Example values for Duration feature
# #     'Price': [200, 300, 250],           # Example values for Price feature
# #     'Employee': ['Alice', 'Bob', 'Charlie']  # Example values for Employee feature
# # }
# # data ka format


@app.route('/predict-sales', methods=['POST'])
def predict2():
    data = request.get_json()
    result = sales.predict2(data.get('days'))
    return jsonify(result)