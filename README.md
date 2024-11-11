# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages (numpy, pandas, SVC, etc).
2. Load the dataset and vectorize text information using CountVectorizer()
3. Fit the data into the model.
4. Check Accuracy Score and Confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Etta supraja
RegisterNumber: 212223220022
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

```
```
data.head()

```
### Output:
![image](https://github.com/user-attachments/assets/d947ecbc-7bf4-4dcb-ac61-c841c36323d0)
```
data.tail()

```
### Output:
![image](https://github.com/user-attachments/assets/5909a894-fda7-4164-99a0-9c2ce1305fd2)

```
data.info()

```
### Output:
![image](https://github.com/user-attachments/assets/975a71ee-66b0-41ae-9c35-9fe211ff3c8c)

```
data.isnull().sum()

```
### Output:
![image](https://github.com/user-attachments/assets/e26032ab-2a32-466c-8b92-4a947099b6ca)

```
x=data['v2'].values

```
```
y=data['v1'].values

```
```
y.shape
```
### Output:
![image](https://github.com/user-attachments/assets/0f679d6f-7573-4a04-8bc2-b89c7af48476)


```
x.shape

```
### Output:
![image](https://github.com/user-attachments/assets/c683dbb2-917e-45c1-938a-35cf4f890816)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

```
```
x_train.shape


```
### Output:
![image](https://github.com/user-attachments/assets/10fa1fe6-6dd5-46d1-a6dc-d18bbcd1f5b8)
```
y_train.shape


```
### Output:
![image](https://github.com/user-attachments/assets/49ecf681-544b-4073-a4e9-dd48acdfd08d)

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


```
```
x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)


```
```
x_train.shape


```
### Output:
![image](https://github.com/user-attachments/assets/40a93a45-4f6b-4e6c-a948-86b104e90763)

```
type(x_train)

```
### Output:
![image](https://github.com/user-attachments/assets/720b5312-c50d-41d7-a18e-bad5cc527946)

```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


```
### Output:
![image](https://github.com/user-attachments/assets/4acb86ea-c30a-45b0-ad73-db7e02fa41f9)

```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```
### Output:
![image](https://github.com/user-attachments/assets/f37e401e-9e82-4e71-8452-5acbe583f578)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
