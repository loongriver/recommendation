import tensorflow as tf 
import numpy as np 
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Embedding,Dense
from tensorflow.keras.models import Model

# load data
data = np.load('train.npy')
u_data = data[:,0]
i_data = data[:,1]
labels = data[:,2]

# 超参数

MAX_R = 5  #最大评分数

# 用户和物品输入
u_input = tf.keras.Input(shape=(1,), dtype='int32', name='u_input')
i_input = tf.keras.Input(shape=(1,), dtype='int32', name='i_input')

# user模型构建
u_x = Embedding(output_dim=32, input_dim=944, input_length=1)(u_input)
u_x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(u_x)
u_x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(u_x)
u_x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(u_x)
u_output = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(u_x)
# item模型构建
i_x = Embedding(output_dim=32, input_dim=1683, input_length=1)(i_input)
i_x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(i_x)
i_x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(i_x)
i_x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(i_x)
i_output = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(i_x)

# 级联之后的层数
x = layers.concatenate([u_output, i_output])
x = layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
# x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
# x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
y_output = layers.Dense(1, activation='sigmoid')(x)



# 计算余弦相似度
# u_i = tf.reduce_sum(tf.multiply(u_output, i_output),axis = 1)
# u_norm = tf.sqrt(tf.reduce_sum(tf.square(u_output), axis = 1))
# i_norm = tf.sqrt(tf.reduce_sum(tf.square(i_output), axis = 1))
# y_ = u_i/tf.multiply(u_norm ,i_norm)
# y_output = Dense(1, activation=None)(y_)


model = Model(inputs=[u_input, i_input], outputs=y_output)


# 定义归一化二分类交叉熵损失函数
def norm_loss(y_true, y_pred):
	return tf.reduce_mean(-y_true * tf.log(y_pred + 1e-10)/MAX_R - (1.0 - y_true/MAX_R) * tf.log(1.0 - y_pred + 1e-10))

model_path = './weights/my_model'

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), loss=norm_loss)
model.summary()

model.fit([u_data, i_data], labels, validation_split=0.2,epochs = 5, batch_size=64)
# model.save_weights(model_path)

# model.load_weights(model_path)


# 定义点击率评价函数
def HitRatio():
	test_data = np.load('test.npy')
	target = np.loadtxt('target.txt', dtype = int)

	print(test_data[1])
	print(test_data[2])
	u_test_data = target[:, 0]
	i_test_data = target[:, 1]
	# u_i_map = {}

	# for u,v in test_data:
	# 	if u not in u_i_map:
	# 		u_i_map[u] = []
	# 	u_i_map[u].append(v)

	# u_rank = {}
	# for key, val in u_i_map.items():
	# 	u_data = np.full((101), key)
	# 	i_data = np.array(val)
	# 	result = model.predict([u_data, i_data])
		# print(result[6])

	result = model.predict([u_test_data, i_test_data], batch_size =32)

	print(result)

HitRatio()
