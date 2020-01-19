from pandas import DataFrame
from sklearn import linear_model
import tkinter as tk
import statsmodels.api as sm

Traffic= {
    'Count': [0, 5, 10, 13, 15, 20, 25, 30, 33, 35, 40, 42, 45, 50, 60, 65, 70, 75,
             80],
    'Delay': [2,3,5,6,6,6,6,10,10,10,12,13,14,15,17,17,19,20,20] }

df = DataFrame(Traffic, columns=['Count', 'Delay'])

X = df[['Count',
        ]]  # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Delay']  # output variable (what we are trying to predict)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=1200, height=450)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(260, 240, window=label_Coefficients)

# with statsmodels
print_model = model.summary()
label_model = tk.Label(root, text=print_model, justify='center', relief='solid', bg='LightSkyBlue1')
canvas1.create_window(800, 220, window=label_model)

# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Enter Count: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
# label2 = tk.Label(root, text=' Type Unemployment Rate: ')
# canvas1.create_window(120, 120, window=label2)
#
# entry2 = tk.Entry(root)  # create 2nd entry box
# canvas1.create_window(270, 120, window=entry2)


def values():
    global New_Count  # our 1st input variable
    New_Count= float(entry1.get())


    # global New_Unemployment_Rate  # our 2nd input variable
    # New_Unemployment_Rate = float(entry2.get())

    Prediction_result = ('Predicted Delay: ', int(regr.predict([[New_Count]])+1))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)

button1 = tk.Button(root, text='Predict Delay', command=values,
                    bg='orange')  # button to call the 'values' command above
canvas1.create_window(270, 150, window=button1)


root.mainloop()