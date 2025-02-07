# %%
import pandas as pd
import seaborn as sns
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
# %%
data = pd.read_csv(r"C:\Turk\Temp\Python\Course BH\Data Set\diabetes.csv")
# %%
data.head()
# %%
data.info()
# %%
data.corr(numeric_only=True)
# %%
data.shape
# %%
data.columns
# %%
data.head()
# %%
data["Outcome"].value_counts()
# %%
data.corr(numeric_only=True)
#%%
data.drop("SkinThickness",axis=1,inplace=True)
data.drop("BMI",axis=1,inplace=True)
#%%
sns.pairplot(data)
#%%
sns.heatmap(data.corr(numeric_only=True),annot=True)
# %%
data.info()
# %%
sns.countplot(x=data["Age"])
# %%
x=data.drop(["Outcome"],axis=1)
y=data["Outcome"]
# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.02,random_state=0)
# %%
model = LogisticRegression()
# %%
model.fit(x_train,y_train)
# %%
y_pred = model.predict(x_test)
# %%
accuracy_score(y_test,y_pred)*100
# %%
recall_score(y_test,y_pred)*100
# %%
f1_score(y_test,y_pred)*100
# %%
confusion_matrix(y_test,y_pred)