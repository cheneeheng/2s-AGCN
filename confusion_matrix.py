from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

label = open('./data/data/ntu/xview/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./data/model/211116110001/epoch50_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
right_num = total_num = right_num_5 = 0
r_list = []
l_list = []
for i in tqdm(range(len(label[0]))):
    _, label_i = label[:, i]
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
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()
