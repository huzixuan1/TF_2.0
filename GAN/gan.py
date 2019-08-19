import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # z:[b,100]-->[b,3*3*512]-->[b,3,3,512]-->[b,64,64,3]
        self.fc=keras.layers.Dense(3*3*512)

        self.conv1=keras.layers.Conv2DTranspose(256,3,3,'valid')  # 反卷积
        self.bn1=keras.layers.BatchNormalization()

        self.conv2=keras.layers.Conv2DTranspose(128,5,2,'valid')
        self.bn2=keras.layers.BatchNormalization()

        self.conv3=keras.layers.Conv2DTranspose(3,4,3,'valid')

    def call(self, inputs, training=None, mask=None):
        # [z,100]-->[z,3*3*512]
        x=self.fc(inputs)
        x=tf.reshape(x,[-1,3,3,512])
        x=tf.nn.leaky_relu(x)

        x=tf.nn.leaky_relu(self.bn1(self.conv1(x),training=training))
        x=tf.nn.leaky_relu(self.bn2(self.conv2(x),training=training))
        x=self.conv3(x)
        x=tf.tanh(x)

        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3]-->[b,1]
        self.conv1=keras.layers.Conv2D(64,5,3,'valid')

        self.conv2=keras.layers.Conv2D(128,5,3,'valid')
        self.bn2=keras.layers.BatchNormalization()

        self.conv3=keras.layers.Conv2D(256,5,3,'valid')
        self.bn3=keras.layers.BatchNormalization()

        # [b,h,w,c]-->[b,-1]
        self.flatten=keras.layers.Flatten()
        # [b,-1]-->[b,1]
        self.fc=keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x=tf.nn.leaky_relu(self.conv1(inputs))
        x=tf.nn.leaky_relu(self.bn2(self.conv2(x),training=training))
        x=tf.nn.leaky_relu(self.bn3(self.conv3(x),training=training))

        x=self.flatten(x)
        logits=self.fc(x)

        return logits

def main():
    d=Discrimination()
    g=Generator()

    x=tf.random.normal([2,64,64,3])
    z=tf.random.normal([2,100])

    prob=d(x)
    print(prob.shape)
    x_hat=g(z)
    print(x_hat.shape)
if __name__ == '__main__':
    main()
