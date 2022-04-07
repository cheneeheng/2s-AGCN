from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

# LABEL = './data/data/ntu/xview/val_label.pkl'
# LABEL = './data/data/ntu_nopad/xview/val_label.pkl'

# BASE = f'data/data/ntu_result/xview'
# FOLDER = f'211228140001_nopad_3ks'
# SCORE = f'{BASE}/aagcn_v17_joint/{FOLDER}/epoch50_test_score.pkl'

# BASE = f'data/data/ntu_result/xview'
# FOLDER = f'220126153001_nopad_3ks_noaug_lowerlr_emd_128d'
# SCORE = f'{BASE}/aagcn_v28_joint/{FOLDER}/epoch50_test_score.pkl'

# label = open(LABEL, 'rb')
# label = np.array(pickle.load(label))
# r1 = open(SCORE, 'rb')
# r1 = list(pickle.load(r1).items())

# right_num = total_num = right_num_5 = 0
# r_list = []
# l_list = []
# for i in tqdm(range(len(label[0]))):
#     _, label_i = label[:, i]
#     _, ranks_i = r1[i]
#     rank_5 = ranks_i.argsort()[-5:]
#     right_num_5 += int(int(label_i) in rank_5)
#     ranks_i = np.argmax(ranks_i)
#     r_list.append(ranks_i)
#     l_list.append(int(label_i))
#     right_num += int(ranks_i == int(label_i))
#     total_num += 1
# acc = right_num / total_num
# acc5 = right_num_5 / total_num
# print(acc, acc5)


# plt.figure(figsize=(30, 10))
# cf_matrix = confusion_matrix(l_list, r_list)
# ax = sns.heatmap(cf_matrix, annot=True, annot_kws={'fontsize': 6}, cmap='jet')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')
# # ## Ticket labels - List must be in alphabetical order
# # ax.xaxis.set_ticklabels(['False','True'])
# # ax.yaxis.set_ticklabels(['False','True'])
# # ## Display the visualization of the Confusion Matrix.
# # plt.show()
# plt.savefig(f'sandbox_results/confmat_{FOLDER}.png', bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(30, 10))
# # per_class_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
# per_class_acc = confusion_matrix(l_list, r_list, normalize='true').diagonal()
# plt.bar([i+1 for i in range(60)], per_class_acc*100.0)
# plt.xlabel('Classes')
# plt.ylabel('Accuracy %')
# plt.ylim(50.0, 100.0)
# plt.savefig(f'sandbox_results/accperclass_{FOLDER}.png', bbox_inches='tight')
# plt.close()


LABEL = './data/data/ntu_sgn/processed_data/NTU_CV_test_label.pkl'
BASE = f'data/data/ntu_result/xview'

# FOLDER = f'220404180001_motionsampler_rerun'
# SCORE = f'{BASE}/sgn_v4/{FOLDER}/epoch107_val_score.pkl'
FOLDER = f'220404120001_rerun_noinit'
SCORE = f'{BASE}/sgn_v4/{FOLDER}/epoch101_val_score.pkl'

label = open(LABEL, 'rb')
label = np.array(pickle.load(label))
r1 = open(SCORE, 'rb')
r1 = list(pickle.load(r1).items())

right_num = total_num = right_num_5 = 0
r_list = []
l_list = []
for i in tqdm(range(len(label))):
    label_i = label[i]
    _, ranks_i = r1[i]
    rank_5 = ranks_i.argsort()[-5:]
    right_num_5 += int(int(label_i) in rank_5)
    ranks_i = np.argmax(ranks_i)
    r_list.append(ranks_i)
    l_list.append(int(label_i))
    right_num += int(ranks_i == int(label_i))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)


plt.figure(figsize=(30, 10))
cf_matrix = confusion_matrix(l_list, r_list)
ax = sns.heatmap(cf_matrix, annot=True, annot_kws={'fontsize': 6}, cmap='jet')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.savefig(f'sandbox_results/confmat_{FOLDER}.png', bbox_inches='tight')
plt.close()

plt.figure()
# per_class_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
per_class_acc = confusion_matrix(l_list, r_list, normalize='true').diagonal()
plt.bar([i for i in range(60)], per_class_acc*100.0)
plt.xlabel('Classes')
plt.ylabel('Accuracy %')
plt.ylim(50.0, 100.0)
plt.savefig(f'sandbox_results/accperclass_{FOLDER}.png', bbox_inches='tight')
plt.close()
