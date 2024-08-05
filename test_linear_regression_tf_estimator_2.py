import tensorflow as tf
import numpy as np
import pandas as pd




# Load the saved model
export_dir = b"export_model\\1722780446"  # เปลี่ยนให้ตรงกับที่คุณบันทึกโมเดล
model_loaded = tf.saved_model.load(export_dir)

# Function to convert data to tf.train.Example
def create_tf_example(data):
    features = {
        'area': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['area'])])),
        'bedrooms': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['bedrooms'])])),
        'bathrooms': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['bathrooms'])])),
        'stories': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['stories'])])),
        'mainroad': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['mainroad'].encode('utf-8')])),
        'guestroom': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['guestroom'].encode('utf-8')])),
        'basement': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['basement'].encode('utf-8')])),
        'hotwaterheating': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['hotwaterheating'].encode('utf-8')])),
        'airconditioning': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['airconditioning'].encode('utf-8')])),
        'parking': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['parking'])])),
        'prefarea': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['prefarea'].encode('utf-8')])),
        'furnishingstatus': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['furnishingstatus'].encode('utf-8')])),
        'price': tf.train.Feature(float_list=tf.train.FloatList(value=[float(data['price'])]))  # Add 'price' if required
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

# Function to predict using the loaded model
def predict(data):
    serialized_example = create_tf_example(data)
    
    # Predict
    predict_fn = model_loaded.signatures["predict"]
    predictions = predict_fn(examples=tf.constant([serialized_example]))

    return predictions

# Example usage
new_data = {
    'area': 7420,
    'bedrooms': 4,
    'bathrooms': 2,
    'stories': 3,
    'mainroad': 'yes',
    'guestroom': 'no',
    'basement': 'no',
    'hotwaterheating': 'no',
    'airconditioning': 'yes',
    'parking': 2,
    'prefarea': 'yes',
    'furnishingstatus': 'furnished',
    'price': 500000  # Add 'price' if the model expects this feature
}


df =pd.read_csv(r"D:\machine_learning_AI_Builders\ML_Algorithm\Linear_regression\Housing.csv")
df["price"] = df["price"].astype("float32")

all_class = df.columns.tolist()

test_data = {key:df[key][500] for key in all_class}
#dict_["price"] = np.log(dict_["price"])
predictions = predict(test_data)
print(predictions)
print(f'predict : {(np.array(predictions["predictions"]).item())} ||| labels : {test_data["price"]}')

#print(f'predict : {np.exp(np.array(predictions["predictions"]).item())} ||| labels : {(dict_["price"])}')



      
