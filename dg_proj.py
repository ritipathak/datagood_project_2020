#sources: https://realpython.com/python-gui-tkinter/
import tkinter as tk
#necessary imports
import numpy as np 
import pandas as pd 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#getting the data
url = 'https://github.com/millenopan/DGMI-Project/blob/master/insurance.csv?raw=true'
data = pd.read_csv(url)


#feature engineering for sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
#feature engineering for smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

#predict smoker 
X = data.drop(['smoker', 'region', 'charges'], axis = 1)
y = data.smoker

#splitting the data into training and test data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=83)



#fitting using lin reg
lreg = LinearRegression()
lreg.fit(X_train, y_train)



#fitting and predicting using random forest
rf = RandomForestRegressor(n_estimators = 200, n_jobs = -1) 
# n_estimators = 200 means 200 trees, n_jobs = -1 uses all your CPU cores to compute them
rf.fit(X_train,y_train)
rf_pred_train = rf.predict(X_train)
rf_pred_test = rf.predict(X_test)



#creating tkinter 

page = tk.Tk()

#creating the widgets 
#title widget 
title = tk.Label(text="I can predict if you are a smoker!", foreground="#FC84FF", background="black", font=(None, 25))


age_label = tk.Label(text="Your Age", foreground='#FC84FF', background="black", font=(None, 15))
age_entry = tk.Entry()
#age = age_entry.get()

sex_label = tk.Label(text="Your Sex(0 for female and 1 for male)", foreground="#FC84FF", background="black", font=(None, 15))
sex_entry = tk.Entry()
#sex = sex_entry.get()

bmi_label = tk.Label(text="Your BMI", foreground="#FC84FF", background="black", font=(None, 15))
bmi_entry = tk.Entry()
#bmi = bmi_entry.get()

children_label = tk.Label(text="Number of Children", foreground="#FC84FF", background="black", font=(None, 15))
children_entry = tk.Entry()

empty_label = tk.Label(text="", background="black")

#print(age)

#our predictor function 
def predictor():  
    age = age_entry.get()
    sex = sex_entry.get()
    bmi = bmi_entry.get()
    children = children_entry.get()
    answer_txt = "You are not a smoker!"
    answer = np.round(rf.predict([[float(age), float(sex), float(bmi), float(children)]]))
    if (answer == 1):
        answer_txt = "You are a smoker!"
    label1 = tk.Label(text= answer_txt, foreground="#FC84FF", background="black", font=(None, 20))
    label1.pack()


#predict button 
button = tk.Button(
    text="Predict!",
    width=4,
    height=1,
    bg="blue", 
    foreground="#DF05E4", 
    command=predictor,
    font=(None, 18)
    #bg="blue",
    #fg="yellow",
)


#source: https://www.daniweb.com/programming/software-development/threads/66181/center-a-tkinter-window
def center_window(w=300, h=350):
    # get screen width and height
    ws = page.winfo_screenwidth()
    hs = page.winfo_screenheight()
    # calculate position x, y
    x = (ws/2) - (w/2)    
    y = (hs/2) - (h/2)
    page.geometry('%dx%d+%d+%d' % (w, h, x, y))

#packing all the widgets 
title.pack()

age_label.pack()
age_entry.pack()
sex_label.pack()
sex_entry.pack()
bmi_label.pack()
bmi_entry.pack()
children_label.pack()
children_entry.pack()
#empty_label.pack()
button.pack()

#starting up page
page.configure(bg='black')
center_window(500, 300) 
page.mainloop()
