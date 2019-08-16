import os
import tensorflow as tf
from Resnet import resnet18
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
tf.random.set_seed(2345)


def preprocess(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255.-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
y_train=tf.squeeze(y_train,axis=1)
y_test=tf.squeeze(y_test,axis=1)
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.shuffle(1000).map(preprocess).batch(64)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_data=test_data.map(preprocess).batch(64)

sample=next(iter(train_data))
print('sample:',sample[0].shape,sample[1].shape,
      tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))

def main():
    model=resnet18()
    model.build(input_shape=(None,32,32,3))
    model.summary()
    optimizer=optimizers.Adam(lr=1e-3)
    for epoch in range(50):
        for step,(x,y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits=model(x)
                y_onehot=tf.one_hot(y,depth=10)
                loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step%100==0:
                print(epoch,step,'loss',float(loss))
        total_num=0
        total_correct=0
        for x,y in test_data:
            logits=model(x)
            prob=tf.nn.softmax(logits,axis=1)
            pred=tf.argmax(prob,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct=tf.reduce_sum(correct)
            total_num+=x.shape[0]
            total_correct+=int(correct)
        acc=total_correct/total_num
        print(epoch,'acc:',acc)
if __name__ == '__main__':
    main()



