#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
# %%
data=pd.read_csv(r"C:\Turk\Temp\Python\Course BH\Data Set\insurance.csv")
# %%
data.head()
# %%
data.info()
# %%
data.corr(numeric_only=True)
# %%
data.shape
# %%
data["sex"].unique()
# %%
data["smoker"].unique()
# %%
data["region"].unique()
# %%
data.drop(["region"],axis=1,inplace=True)
# %%
data.columns
# %%
data["sex"].value_counts()
# %%
data["sex"]=[1 if row=="male" else 0 for row in data["sex"]]
# %%
data["sex"].value_counts()
# %%
data["smoker"].value_counts()
# %%
data["smoker"]=[1 if row=="no" else 0 for row in data["smoker"]]
# %%
data["smoker"].value_counts()
# %%
data.corr(numeric_only=True)
# %%
sns.countplot(x=data["age"])
# %%
sns.pairplot(data,hue="charges")
# %%
plt.figure(figsize=(24,10))
sns.heatmap(data.corr(),annot=True,mask=data.corr()<0.9,cmap="Blues")
#%%
x=data.drop(["charges"],axis=1)
y=data["charges"]
# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.99,random_state=0)
# %%
model=LinearRegression()
# %%
model.fit(x_train,y_train)
# %%
data.info()
# %%
y_pred = model.predict(x_test)
# %%
y_pred
# %%
x_test
# %%
y_test
# %%
mean_absolute_error(y_test,y_pred)
# %%
r2_score(y_test,y_pred)*100