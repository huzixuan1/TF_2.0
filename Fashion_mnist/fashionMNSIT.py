import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
# 减少TF打印出多的信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# train x,y
# test  x,y
(x_train,y_train),(x_test,y_test)=datasets.fashion_mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
def preprocess(x,y):
    x=tf.cast(x,dtype=tf.float32)/255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y
batchsz=128
# build dataset
train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
# deal datasets
train_data=train_data.map(preprocess).shuffle(10000).batch(batchsz)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.map(preprocess).batch(batchsz)

# print sample
train_iter=iter(train_data)
sample=next(train_iter)
print('batch:',sample[0].shape,sample[1].shape)

# build model
model=Sequential([
    # full layers:Dense
    layers.Dense(256,activation=tf.nn.relu),  # [b,784]-->[b,256]
    layers.Dense(128,activation=tf.nn.relu),  # [b,128]-->[b,64]
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(16,activation=tf.nn.relu),
    layers.Dense(10)
])
# input data to build weight
model.build(input_shape=[None,28*28])
model.summary()


# define optimzer
# w=w-lr*grad
optimizer=optimizers.Adam(lr=1e-3)

def main():
    # forward
    for epoch in range(30):
        for step,(x_train,y_train) in enumerate(train_data):
            x=tf.reshape(x_train,[-1,28*28])
            with tf.GradientTape() as tape:
                logits=model(x)
                y_onehost=tf.one_hot(y_train,depth=10)
                loss_mse=tf.reduce_mean(tf.losses.MSE(y_onehost,logits))
                loss_ce=tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehost,logits,from_logits=True))
            grads=tape.gradient(loss_ce,model.trainable_variables)

            # update,zip:connect
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step%100==0:
                print(epoch,step,'loss:',float(loss_ce))
        # total number
        total_corr=0
        total_num=0
        for x_test,y_test in test_data:
            x=tf.reshape(x_test,[-1,28*28])
            logits=model(x)
            prob=tf.nn.softmax(logits,axis=1)
            predic=tf.argmax(prob,axis=1)
            predic=tf.cast(predic,dtype=tf.int32)
            correct=tf.equal(predic,y_test)
            correct=tf.reduce_sum(tf.cast(correct,dtype=tf.int32))
            total_corr+=int(correct)
            total_num+=x.shape[0]
        acc=total_corr/total_num
        print(epoch,'test acc:',acc)
if __name__ == '__main__':
    main()