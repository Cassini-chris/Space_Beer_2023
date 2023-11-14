from flask import Flask
from flask import jsonify
import numpy as np
from flask import request, render_template
import io
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from datetime import datetime, timedelta
import os
from google.cloud import storage
import tempfile
from random import *
import tensorflow as tf
from tensorflow.keras.models import load_model

from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import sqlalchemy

app = Flask(__name__)

@app.route('/')
def index():
    #return("test")
    return render_template('web-interface.html')

#Download model from gcloud bucket
storage_client = storage.Client()
bucket_name = '****'

model = 'beer_model_v5.h5'
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model)
temp_path = "/tmp/beer_model_v5.h5"
blob.download_to_filename(temp_path)

model_base = 'base_model_v2.h5'
bucket = storage_client.get_bucket(bucket_name)
blob_base = bucket.blob(model_base)
temp_path_base = "/tmp/base_model_v2.h5"
blob_base.download_to_filename(temp_path_base)

# Get the connection details from the environment variables
db_user = '****'
db_pass = '****'
db_name = '****'
db_table = '****'
instance_connection_name = '****'

print("Start SQL")

def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool

pool = connect_with_connector()
print("Start Pool")

proxy = conn.execute(text('SELECT * FROM '****''))
resultproxy = pool.execute('SELECT * FROM '****'')

d, a = {}, []
for rowproxy in resultproxy:
    # rowproxy.items() returns an array like [(key0, value0), (key1, value1)]
    for column, value in rowproxy.items():
        # build up the dictionary
        d = {**d, **{column: value}}
    a.append(d)

@app.route("/_ah/warmup")
def warmup():
    """Served stub function returning no content.
    Your warmup logic can be implemented here (e.g. set up a database connection pool)
    Returns:
    An empty string, an HTTP code 200, and an empty object.
    """
    return "", 200, {}

def get_model():
    global model
    model = tf.keras.models.load_model(temp_path)
    print(" * Beer-Model loaded--u!")
    return model

def get_model_base():
    global model_base
    model_base = tf.keras.models.load_model(temp_path_base)
    print(" * Base-Model loaded--!")
    return model_base

print(" * Loading Keras Model")
get_model()
get_model_base()

@app.route("/predicto", methods = ["POST"])
def predicto():
    message = request.get_json(force=True)
    print("Message: ", message)
    encoded = message[0]['image']

    encoded_bytes = str.encode(encoded)
    now = datetime.now() + timedelta(hours=2)

    encoded += "=="
    decoded = base64.b64decode(encoded)
    print("Decoded: ", decoded)

    img_check = encoded[ 0 : 4 ]
    if img_check == "/9j/":
        img_check = "jpg"
    else:
        img_check = "png"

    print("space_beer_uploads: ")
    bucket_name = 'space_beer_uploads'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("space_beer/upload_folder/upload_" + str(now) + "." + img_check)
    blob.upload_from_string(decoded, content_type='image/' + img_check)

    #Convert decoded into Tensor
    image = tf.image.decode_image(decoded, channels=3)

    #Convert to Float
    image = tf.image.convert_image_dtype(image, tf.float32)

    #Resize Tensor
    image = tf.image.resize(image, [224, 224])

    #Expanding (1, 224, 224, 3)
    processed_image = np.expand_dims(image, axis=0)

    preds = model_base.predict(processed_image)

    from keras.applications.mobilenet_v2 import decode_predictions

    beer_check = False

    for name, desc, score in decode_predictions(preds)[0]:
        print('- {} ({:.2f}%%)'.format(desc, 100 * score))

    for name, desc, score in decode_predictions(preds)[0]:
        if (desc=='beer_glass'):
            print("yes")
            beer_check = True

        else:
            print("no beer detected")

    if (beer_check == True):
        prediction  =  model.predict(processed_image).tolist()
        print("Prediction: ", prediction)
        prediction_result = prediction[0][0]
        perfect_result = prediction[0][1]
        print("Prediction Result: ", prediction_result)

        ## Option for Heavy TOP:
        ## normalized_prediction_with_heavy_top = ((prediction[0][0]) * 1 +  (prediction[0][2]) * 2 + (prediction[0][3]) * 3 + (prediction[0][4]) * 4 + (prediction[0][5]) * 5 + (prediction[0][6]) * 6.5 + (prediction[0][7]) * 7.5 + (prediction[0][8]) * 8.5 + (prediction[0][9]) * 9 + (prediction[0][1]) * 10) / 10

        normalized_prediction = ((prediction[0][0]) * 1 +  (prediction[0][2]) * 2 + (prediction[0][3]) * 3 + (prediction[0][4]) * 4 + (prediction[0][5]) * 5 + (prediction[0][6]) * 6 + (prediction[0][7]) * 7 + (prediction[0][8]) * 8 + (prediction[0][9]) * 9 + (prediction[0][1]) * 10) / 10

        print("#####################################################")
        print("normalized_prediction:" , normalized_prediction)
        perfect_result = normalized_prediction
        print("#####################################################")

        now2 = now.strftime("%Y-%m-%d %H:%M:%S")
        perfect_result_for_db = perfect_result * 100
        string_perfect_result =str(perfect_result_for_db)

        name = message[0]['name']
        if (name==""):
            name="Average Gustav"

        ins = ("INSERT INTO '****' (datetime, score, name) VALUES(%(datetime)s, %(score)s, %(name)s)")
        input_score = {
            'datetime': now2,
            'score': string_perfect_result,
            'name': name,
        }

        pool.execute(ins, input_score)
        resultproxy = pool.execute('WITH space_beer_results AS ((SELECT *, 1 as is_you FROM '****'  ORDER BY datetime DESC LIMIT 1 )UNION ALL (SELECT *, 0 as you From '****'  ORDER BY DATETIME DESC LIMIT 999 OFFSET 1)) SELECT *, RANK () OVER (ORDER BY SCORE DESC ) rank_no  FROM space_beer_results ORDER by SCORE DESC LIMIT 1000')

        d, a = {}, []
        for rowproxy in resultproxy:
            # rowproxy.items() returns an array like [(key0, value0), (key1, value1)]
            for column, value in rowproxy.items():
                # build up the dictionary
                d = {**d, **{column: value}}
            a.append(d)

        # Placeholder for two classes - To be rewritten
        response = {'prediction': {
                'level_1':prediction[0][0],
                'level_10':perfect_result
        },
        'database': a
        }
    else:
        response = {'prediction': {
                        'level_1':"No Beer"
        }}

    return jsonify(response)
