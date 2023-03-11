import numpy as np
from read_ import *
from image_process import HOG
import math
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from  sklearn.metrics import accuracy_score
# from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
from top_1_top_5 import top1_accuracy,top5_accuracy
import matplotlib.pyplot as plt


def per2lab(praba):
    label = []
    for pra in praba:
        label.append(np.where(pra == np.max(pra)))
    label = np.asarray(label).flatten()

    return label

train_inf = read_local_txt("train")
vail_inf = read_local_txt("val")
test_inf = read_local_txt("test")

batch_size = 500
train_len = train_inf.shape[0]
train_class =train_inf.iloc[:,1].unique()
sum_ba = 0
# train_inf1 = train_inf.iloc[:,0]

sgd = SGDClassifier(loss = "log",penalty="elasticnet")
per = Perceptron(penalty = "elasticnet")
# lgb = LGBMClassifier(boosting_type = "gbdt",objective = "multiclass",keep_training_booster=True)

per_train_acc = []
sgd_train_acc = []
gbm_train_acc = []

per_vali_acc = []
sgd_vali_acc = []
gbm_vali_acc = []



vail_images = read_image(vail_inf.iloc[:,0])

vail_img =  HOG(vail_images)
vail_img = np.asarray(vail_img)
vail_label = vail_inf.iloc[:,1]

test_images = read_image(test_inf.iloc[:,0])

test_img =  HOG(test_images)
test_img = np.asarray(test_img)
test_label = vail_inf.iloc[:,1]
params={
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'seed': 2021,
        'num_class': 50,
        "num_iterations" : 100,
        'verbose': -1,
        "num_iterations" : 10000,
        "objective" : "multiclass",
        'metric': 'multi_error'
    }


for n in range(math.ceil(train_len/batch_size)):

    batch,label,train_inf,batch_size1 = train_batch(train_inf,batch_size = batch_size)
    train_imges = read_image(batch)
    sum_ba += batch_size1

    train_img = HOG(train_imges)
    train_img = np.asarray(train_img)

    lgb_train = lgb.Dataset(train_img, label.values)
    lgb_eval = lgb.Dataset(vail_img, vail_label.values, reference=lgb_train)


    if n <1 :
        sgd.partial_fit(train_img,label,classes=train_class)
        per.partial_fit(train_img,label,classes=train_class)
        # lgb.partial_fit(train_img, label, classes=train_class)

    else:
        sgd.partial_fit(train_img, label)
        per.partial_fit(train_img, label)
        # lgb.partial_fit(train_img, label)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=None,  # 如果gbm不为None，那么就是在上次的基础上接着训练
                    callbacks=[log_evaluation(0),early_stopping(200,verbose=False)],
                    keep_training_booster=True,
                    # force_col_wise=True
                    )

    sgd_score = sgd.score(train_img, label)
    per_score = per.score(train_img, label)
    gbm_score = accuracy_score(per2lab(gbm.predict(train_img)),label)

    sgd_vali = sgd.predict(vail_img)
    per_vali = per.predict(vail_img)
    gbm_vail = per2lab(gbm.predict(vail_img))

    sgd_vali_accuracy = accuracy_score(vail_label,sgd_vali)
    per_vali_accuracy = accuracy_score(vail_label,per_vali)
    gbm_vail_accuracy = accuracy_score(vail_label,gbm_vail)


    sgd_train_acc.append(sgd_score)
    per_train_acc.append(per_score)
    gbm_train_acc.append(gbm_score)

    sgd_vali_acc.append(sgd_vali_accuracy)
    per_vali_acc.append(per_vali_accuracy)
    gbm_vali_acc.append(gbm_vail_accuracy)

    print("-------" + str(sum_ba) + "/" + str(train_len) + "------" + "sgd_train_acc:" + str(sgd_score) + " // sgd_vali_accuracy:" + str(sgd_vali_accuracy))
    print("-------" + " // per_train_acc:" + str(per_score) + " // per_vali_accuracy:" +str(per_vali_accuracy))
    print("-------" + " // gbm_train_acc:" + str(gbm_score) + " // gbm_vali_acc:" + str(gbm_vail_accuracy))




print("Perceptron----" + "top_1_acc : " + str(top1_accuracy(test_label,per.decision_function(test_img))) + "top_5_acc : " + str(top5_accuracy(test_label,per.decision_function(test_img))))
print("SGD-------" + "top_1_acc : " + str(top1_accuracy(test_label,sgd.predict_proba(test_img))) + "top_5_acc : " + str(top5_accuracy(test_label,sgd.predict_proba(test_img))))
print("LightGBM-------" + "top_1_acc : " + str(top1_accuracy(test_label,gbm.predict(test_img))) + "top_5_acc : " + str(top5_accuracy(test_label,gbm.predict(test_img))))



#---------------------- plot the graph ----------------------



xlab = range(batch_size,train_len,batch_size)
xlab = list(xlab)
if batch_size*len(xlab)< train_len:
    xlab.append(train_len)

plt.figure(figsize=(8,8))

plt.subplot(131)
plt.plot(xlab,sgd_train_acc,'s-',color = 'r',label="sgd_train_acc")#s-:方形
plt.plot(xlab,sgd_vali_acc,'o-',color = 'g',label="sgd_vail_acc")#o-:圆形
plt.xlabel("epotch")#横坐标名字
plt.ylim((0,1))
plt.ylabel("accuracy")#纵坐标名字
plt.title("SGDClassifier")
plt.legend(loc = "best")#图例

plt.subplot(132)
plt.plot(xlab,per_train_acc,'s-',color = 'r',label="per_train_acc")#s-:方形
plt.plot(xlab,per_vali_acc,'o-',color = 'g',label="per_train_acc")#o-:圆形
plt.ylim((0,1))
plt.xlabel("epotch")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.title("Perceptron")
plt.legend(loc = "best")#图例

plt.subplot(133)
plt.plot(xlab,gbm_train_acc,'s-',color = 'r',label="gbm_train_acc")#s-:方形
plt.plot(xlab,gbm_vali_acc,'o-',color = 'g',label="gbm_vali_acc")#o-:圆形
plt.ylim((0,1))
plt.xlabel("epotch")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.title("LightGBM")
plt.legend(loc = "best")#图例


plt.show()





