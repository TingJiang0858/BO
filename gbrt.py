from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Scale the data to between 0 and 1
X_train = X_train/ 255
X_test = X_test/ 255

#Flatten arrays from (28x28) to (784x1)
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

#Convert the y's to categorical to use with the softmax classifier
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Establish the input shape for our Networks.
input_shape= X_train[0].shape


#imports we know we'll need
import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer  

import pandas as pd

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow
from tensorflow.python.keras import backend as K
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_dense_nodes = Integer(low=1, high=28, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_batch_size = Integer(low=1, high=128, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 1,512, 13, 'relu',64, 1e-3]

from tensorflow.keras.optimizers import Adam


def create_model(learning_rate, num_dense_layers, num_input_nodes,
                 num_dense_nodes, activation, adam_decay):
    # start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(num_input_nodes, input_shape=input_shape, activation=activation
                    ))
    # create a loop making a new dense layer for the amount passed to this model.
    # naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name
                        ))
    # add our classification layer.
    model.add(Dense(10, activation='softmax'))

    # setup our optimizer and compile
    adam = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_input_nodes,
            num_dense_nodes, activation, batch_size, adam_decay):
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_input_nodes=num_input_nodes,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         adam_decay=adam_decay
                         )

    # named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                         y=y_train,
                         epochs=3,
                         batch_size=batch_size,
                         validation_split=0.15,
                         )
    # return the validation accuracy for the last epoch.
    history_dict = blackbox.history
    print(history_dict.keys())
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    print("acc", acc)
    print("val:", val_acc)

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.compat.v1.reset_default_graph()

    # the optimizer aims for the lowest score, so we return our negative accuracy
    acy = min(acc)
    return -acy

gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters
                        )
print("*****===================>")
print("gp:",-gp_result.fun)
ss = pd.DataFrame(gp_result.x_iters)
ss.describe()
'''
model = create_model(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4],gp_result.x[5])
model.fit(X_train,y_train, epochs=3)
print(">>>>>>>>>>>>>>>>>>>>>>>")
print(model.evaluate(X_test,y_test))
'''
gbrt_result = gbrt_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            n_jobs=-1,
                            x0=default_parameters
                            )
print("%%%%%==============>")
print("gbrt:", -gbrt_result.fun)
#print(gbrt_result)
s = pd.DataFrame(gbrt_result.x_iters)
s.describe()
