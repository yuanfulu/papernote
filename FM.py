import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend


# 二阶交叉层
class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMLayer, self).__init__(**kwargs)

    # 初始化训练权重
    def build(self, input_shape):
        self.v = self.add_weight(name='kernel',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(FMLayer, self).build(input_shape)

    # 二阶交叉项的计算公式
    def call(self, x):
        a = K.pow(K.dot(x, self.v), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.v, 2))
        return K.sum(a-b, 1, keepdims=True)*0.5

    # 输出的尺寸大小
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


# 实现FM算法
def FM(feature_dim,out_dim):
    inputs = tf.keras.Input((feature_dim,))
    # 线性回归
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    # 二阶交叉
    cross = FMLayer(feature_dim,out_dim)(inputs)

    # FM（线性回归 + 二阶交叉项）
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model


# 训练FM模型
def train():
    fm = FM(30,4)
    data = load_breast_cancer()

    # sklearn 切分数据
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                        test_size=0.2,random_state=11,
                                                        stratify=data.target)
    fm.fit(X_train, y_train, epochs=5, batch_size=20, validation_data=(X_test, y_test))
    return fm


if __name__ == '__main__':
    fm = train()
    fm.summary()
