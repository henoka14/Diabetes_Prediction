# from django.shortcuts import render

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# def home(request):
#     return render(request, 'home.html')
# def predict(request):
#     return render(request, 'predict.html')
# def result(request):
#     file_path = "/Users/henokabraha/Desktop/Diabetes_Prediction/diabetes.csv"

#      data = pd.read_csv(file_path)
     
#      X = data.drop("Outcome", axis=1)
# Y = data['Outcome']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train


# model = LogisticRegression(max_iter=600)
# model.fit(X_train, Y_train)

#     val1 = float(request.GET['n1'])
#     val2 = float(request.GET['n2'])
#     val3 = float(request.GET['n3'])
#     val4 = float(request.GET['n4'])
#     val5 = float(request.GET['n5'])
#     val6 = float(request.GET['n6'])
#     val7 = float(request.GET['n7'])
#     val8 = float(request.GET['n8'])
    
#     pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8,]])
    
#     result2 = ""
#         if pred == [1]:
#             result2 = "positive"
#         else:
#             result2 = "negative"
            
#         return render(request, "predict.html", {"result2":result}) 


#     return render(request, 'predict.html')

from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load the dataset
    file_path = "/Users/henokabraha/Desktop/Diabetes_Prediction/diabetes.csv"
    data = pd.read_csv(file_path)
    
    # Prepare the data
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression(max_iter=600)
    model.fit(X_train, Y_train)
    
    # Get input values from the request
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    
    # Make a prediction
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    # Interpret the result
    if pred == [1]:
        result2 = "positive"
    else:
        result2 = "negative"
    
    # Render the result page
    return render(request, "predict.html", {"result2": result2})

