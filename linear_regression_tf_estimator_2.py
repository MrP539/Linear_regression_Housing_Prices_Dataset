import pandas as pd
import scipy
import scipy.stats
import numpy as np
import sklearn
import sklearn.model_selection
import tensorflow as tf

df =pd.read_csv(r"D:\machine_learning_AI_Builders\ML_Algorithm\Linear_regression\Housing.csv")

############################################################################  preprocessing data  #####################################################################

all_feature_name = df.columns.tolist()
print(all_feature_name)

# -- check null value
def check_type_and_nav_values(class_list):
    for feature in class_list:

        #print(type(df[feature][0]) == type("A"))
        
        if df[feature].isnull().values.any():
            print(feature)
        else: 
            print("None\n")




check_type_and_nav_values(all_feature_name)


# df_drop = df.drop(labels="Address",axis=1)
# feature_name = [values for values in feature_name if values != "Address"]

# print(df_drop.shape)
# print(feature_name)
print(df.head(5))

# df_drop["Price"] = df_drop.apply(lambda x: x.Price.replace("$","").replace(",","").replace(".00",""),axis=1)
df["price"] = df["price"].astype("float32")

# print(df_drop.head(5))

price = df["price"]
print(f"Max: {price.max()} Min : {price.min()}")

z_score = scipy.stats.zscore(df["price"])
        #  Z-score หรือ Standard Score คือค่าที่บอกถึงจำนวนส่วนเบี่ยงเบนมาตรฐานที่ค่าหนึ่งอยู่ห่างจากค่าเฉลี่ยของกลุ่มข้อมูลนั้นๆ ในทางสถิติ ค่า Z-score คำนวณได้จากสูตร:
        # data-mean/std


filted_entries = np.abs(z_score) <3 #np.abs== absolute value

test = [0 for _ in filted_entries.tolist() if not _]
print(len(test))
# print(filted_entries.tolist())

filted_entries = np.abs(z_score) <3 #np.abs== absolute value

data_feeding = df[filted_entries]

print(f"feeding_data : {data_feeding.shape}")

#targets = np.log(data_feeding["price"])  ## การใช้ log กับ ราคา จะทำให้การ prediction แม่นยำมากขึ้น
targets = (data_feeding["price"])  ## การใช้ log กับ ราคา จะทำให้การ prediction แม่นยำมากขึ้น


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(data_feeding,targets,test_size=0.2,random_state=101,shuffle=True)

print(x_test)

numeric_column = [feature for feature in all_feature_name if type(df[feature][0]) != type("A")]

numeric_feature = [tf.feature_column.numeric_column(key=column) for column in numeric_column]

categorical_column = [feature for feature in all_feature_name if type(df[feature][0]) == type("A")]

categorical_feature = [tf.feature_column.categorical_column_with_vocabulary_list(key=column,vocabulary_list=df[column].unique()) for column in categorical_column]

# linear_feature = numeric_feature
linear_feature = numeric_feature + categorical_feature

# #################################################  create tf.data for feed to model  ###########################################################

def train_input_func(features,target,epochs,shuffle=True,batch_size = 128):
    
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),target))
        if shuffle :
            dataset.shuffle(100)
        dataset = dataset.batch(batch_size=batch_size).repeat(epochs)
        return dataset
    return input_function

def eval_input_func(features,target=None,shuffle=False,batch_size = 128):

    def input_function():
        inputs_features = dict(features)

        if target is not None:
            inputs_feature = (inputs_features,target)
        else:
            inputs_feature = inputs_features
        
        dataset = tf.data.Dataset.from_tensor_slices(inputs_feature)

        if shuffle :
            dataset.shuffle(100)

        dataset = dataset.batch(batch_size=batch_size)
        return(dataset)

    return input_function  


train_input_func_ = train_input_func(x_train,y_train,epochs=None)

LinearRegression_model = tf.estimator.LinearRegressor(feature_columns=linear_feature)

LinearRegression_model.train(input_fn=train_input_func_,steps=2000)


# #################################################################   Evaluate   ############################################################################

eval_input_func_ = eval_input_func(x_test,y_test,shuffle=False)

eval = LinearRegression_model.evaluate(input_fn=eval_input_func_)


print(f"\n\n>> Evaluatetion result : {eval} \n\n")



# #################################################################   Prediction   ############################################################################

# -- all

pred_list = list(LinearRegression_model.predict(input_fn=eval_input_func_))

# target = (np.exp(np.array(y_test))).tolist()
# all_sample = [np.exp(pred_list[sample_unit]['predictions'].item()) for sample_unit in range(len(x_test))]

target = ((np.array(y_test))).tolist()
all_sample = [(pred_list[sample_unit]['predictions'].item()) for sample_unit in range(len(x_test))]

print(f"\n\n>> Prediction result  \n\n")

for i in range(10):
    all_sample_ = all_sample[i]
    target_ = target[i]
    print(f"\n\npredict : {all_sample_:.2f} $  VS Labels : {target_:.2f} $")


# -- unit

sample_unit = 0

# sample = (np.exp(pred_list[sample_unit]['predictions'].item()))
# sample_target = np.exp(np.array(y_test.iloc[sample_unit]))

sample = ((pred_list[sample_unit]['predictions'].item()))
sample_target = (np.array(y_test.iloc[sample_unit]))


print(f"\n\n>> Unit {sample_unit} predict : {sample:.2f} $  VS Labels : {sample_target:.2f} $")

import matplotlib.pyplot as plt
import seaborn as sns

# เปลี่ยน target และ all_sample เป็น numpy array
target = np.array(target)
all_sample = np.array(all_sample)

# สร้าง DataFrame สำหรับการ Plot
import pandas as pd
plot_data = pd.DataFrame({
    'Actual': target,
    'Predicted': all_sample
})

# Plot
plt.figure(figsize=(12, 6))

# Scatter plot
sns.scatterplot(data=plot_data, x='Actual', y='Predicted', alpha=0.5)
plt.plot([plot_data['Actual'].min(), plot_data['Actual'].max()],
         [plot_data['Actual'].min(), plot_data['Actual'].max()],
         color='red', linestyle='--')

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()



############################################################################# Save model  ##########################################################################

feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=linear_feature)

serving_inputs_receiver_func = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=feature_spec)

export_dir = LinearRegression_model.export_saved_model(export_dir_base="export_model",serving_input_receiver_fn=serving_inputs_receiver_func)
print(export_dir)

# ###############################################################################  load model  ##########################################################################

# model_loaded= tf.saved_model.load(export_dir)

# def predict(x):
#     example = tf.train.Example()
#     example.features.feature['price'].float_list.value.extend([x])
#     return model_loaded.signatures["predict"](examples = tf.constant([example.SerializeToString()]))