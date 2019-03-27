import numpy as np 

all_ratings = np.loadtxt('ml-100k/u.data')
all_ratings.astype(int)


# 过滤掉最后一次交互
def filter_latest_interaction(all_ratings):
	user_map_item = {}
	for u, v, r, t in all_ratings:
		if u not in user_map_item:
			user_map_item[u] = {}
			user_map_item[u][v] = t

	latest_item_interaction = {}
	for u in user_map_item:
		item = -1
		time = -1
		for v in user_map_item[u]:
			if(time < user_map_item[u][v]):
				time = user_map_item[u][v]
				item = v
		latest_item_interaction[u] = v
	with open('test.txt', 'w') as f:
		for key, val in latest_item_interaction.items():
			f.write(str(int(key)) +'\t'+ str(int(val)) + '\n')


#  负采样
all_ratings_map_u = {}

for u, v, r, t in all_ratings:
	if u not in all_ratings_map_u:
		all_ratings_map_u[u] = {}
	all_ratings_map_u[u][v] = 1

u_max_num = 943
i_max_num = 1682
neg_ratio = 7
test_samples = 100

def sample_one(all_ratings_map_u):
	# 抽样训练数据
	u_rand_num = np.random.randint(1, u_max_num)
	v_rand_num = np.random.randint(1, i_max_num) 

	if u_rand_num in all_ratings_map_u and v_rand_num not in all_ratings_map_u[u_rand_num]:
		return u_rand_num, v_rand_num
	else:
		return sample_one(all_ratings_map_u)

def sample_test(u_num, all_ratings_map_u):
	# 抽样测试数据
	v_rand_num = np.random.randint(1, i_max_num) 

	if v_rand_num not in all_ratings_map_u[u_num]:
		return v_rand_num
	else:
		return sample_test(u_num, all_ratings_map_u)


# 将原评分数据和抽样数据拼接起来组成训练数据
def get_train_data(all_ratings):
	sample_list = []
	for _ in range(all_ratings.shape[0]):
		for _ in range(neg_ratio):
			u, v = sample_one(all_ratings_map_u)
			sample_list.append([u, v, 0])
			all_ratings_map_u[u][v] = 0

	# 从训练集中删除掉测试数据
	latest_interaction = np.loadtxt('target.txt')
	interaction_map = {}

	for u, v in latest_interaction:
		interaction_map[u] = v

	i = 0
	for u, v, _, _ in all_ratings:
		if u in interaction_map and v == interaction_map[u]:
			all_ratings = np.delete(all_ratings, i, 0)
		i = i+1

	x = np.concatenate((all_ratings[:, 0:3], np.array(sample_list)), axis = 0)
	x.astype(int)
	print(all_ratings.shape)
	print(x.shape)
	permutation = np.random.permutation(x.shape[0])
	shuffled_x = x[permutation, :]
	np.save("train.npy", shuffled_x)

# 生成测试数据
def get_test_data():
	test_list = []
	for u in range(1, u_max_num+1):
		for _ in range(test_samples):
			v = sample_test(u, all_ratings_map_u)
			test_list.append([int(u), int(v)])
			all_ratings_map_u[u][v] = 0

	latest_interaction = np.loadtxt('target.txt', dtype = int)
	x = np.concatenate((latest_interaction, np.array(test_list)), axis = 0)
	x.astype(int)
	print(x.shape)
	np.save("test.npy", x)

get_train_data(all_ratings)