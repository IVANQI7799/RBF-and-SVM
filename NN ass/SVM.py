from scipy.io import loadmat
from sklearn import svm
from sklearn import model_selection

x = loadmat("data_train.mat")["data_train"] #330*33
y = loadmat("label_train.mat")["label_train"]  #330*1
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=1,test_size=0.2)
x_test= loadmat("data_test.mat")["data_test"]
clf = svm.SVC(C=0.8, kernel='rbf', gamma=0.4, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

def show_accuracy(y_hat,y_train,str):
    pass

print("SVM-输出训练集的准确率为：",clf.score(x_train,y_train))
y_hat=clf.predict(x_train)
show_accuracy(y_hat,y_train,'训练集')

y_hat=clf.predict(x_test)
#ture=0
#false=0
#for i in range(66):
    #if(y_hat[i]==y_test[i]): ture+=1
   # else: false+=1
#acc=ture/(ture+false)
#print("SVM-输出验证集的准确率为：")
#print(acc)
print(y_hat)
