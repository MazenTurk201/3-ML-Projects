#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
# %%
df = pd.read_csv(r"C:\Turk\Temp\Python\Course BH\Data Set\USA_Housing.csv")
df.head()
# %%
df.describe()
# %%
df.info()
# %%
df.drop(["Address"],axis=1,inplace=True)
df.info()
# %%
df.corr(numeric_only=True)
# %%
sns.scatterplot(x="Area Population", y="Price",data=df)
# %%
x=df.drop(["Price"],axis=1)
y=df["Price"]
# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
# %%
model=LinearRegression()
# %%
model.fit(x_train,y_train)
# %%
y_pred=model.predict(x_test)
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