# %%
import pandas as  pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("placement dataset.csv")
data

# %%
x = data[['cgpa']]
y = data[['package']]


# %%
x


# %%
y

# %%
plt.title("CGPA vs Package")
plt.xlabel("CGPA")
plt.ylabel("Package")
plt.scatter(x, y)

# %%
from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(x,y,test_size=0.2,random_state=42)

# %%
len(x_train), len(x_test)

# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)


# %%
y_pred = lr.predict(x_test)


# %%
y_test.values

# %%
m = lr.coef_
b = lr.intercept_

test = m*6.94 + b
test

# %%
plt.title("CGPA vs Package")
plt.xlabel("CGPA")
plt.ylabel("Package")
plt.scatter(x, y)
plt.plot(x_test, lr.predict(x_test),color='red')

# %%
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("r2 Score:", r2_score(y_test, y_pred))


# %%
r2 = r2_score(y_test, y_pred)

print("Adjusted r2 Score:", 1-(1-r2)*(len(y_test)-1)/(len(y_test) - 1 - x_test.shape[1]))

## Formula for Adjusted R2 Score = 1 - (1-R2)*(n-1)/(n-1-k)
## where n = number of observations, k = number of independent variables

# %%
cgpa = float(input("Enter your CGPA: "))
x = lr.predict([[cgpa]])
print(f"Your predicted package is: {x}")



