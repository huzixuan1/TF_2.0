import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
def preprocess(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1.
    y=tf.cast(y,dtype=tf.int32)
    return x,y

batchsz=128
# load dataset
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
y_train=tf.squeeze(y_train)
y_test=tf.squeeze(y_test)
y_train=tf.one_hot(y_train,depth=10)
y_test=tf.one_hot(y_test,depth=10)
# print('dateset:',x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_train.min(),x_train.max())

# train data
train_date=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_date=train_date.map(preprocess).shuffle(10000).batch(batchsz)
# test data
test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.map(preprocess).batch(batchsz)

# sample=next(iter(train_date))
# print(sample[0].shape,sample[1].shape)

class MyDense(layers.Layer):
    def __init__(self,inp_dim,output_dim):
        super(MyDense, self).__init__()
        self.kernel=self.add_variable('w',[inp_dim,output_dim])
        # self.bias=self.add_variable('b',[output_dim])

    def call(self,input,training=None):
        x=input@self.kernel
        return x
class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1=MyDense(32*32*3,256)
        self.fc2=MyDense(256,128)
        self.fc3=MyDense(128,64)
        self.fc4=MyDense(64,32)
        self.fc5=MyDense(32,16)
        self.fc6=MyDense(16,10)
    def call(self, input, training=None):
        # reshape
        x=tf.reshape(input,[-1,32*32*3])
        # [b,32*32*3]-->[b,256]
        x=self.fc1(x)
        x=tf.nn.relu(x)
        # [b,256]-->[b,128]
        x=self.fc2(x)
        x=tf.nn.relu(x)
        # [b,64]-->[b,64]
        x=self.fc3(x)
        x=tf.nn.relu(x)
        # [b,64]-->[b,32]
        x=self.fc4(x)
        x=tf.nn.relu(x)
        # [b,32]-->[b,16]
        x=self.fc5(x)
        x=tf.nn.relu(x)
        # [b,16]-->[b,10]
        x=self.fc6(x)
        return x
network=MyNetwork()
network.compile(optimizer=optimizers.Adam(1e-4),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_date, epochs=15, validation_data=test_data, validation_freq=1)


# save model
network.evaluate(test_data)
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')

# create network
network=MyNetwork()
network.compile(optimizer=optimizers.Adam(1e-4),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_date, epochs=15, validation_data=test_data, validation_freq=1)
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')

network.evaluate(test_data)








