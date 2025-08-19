import streamlit as st,pandas as pd,seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
st.sidebar.header('Modify the below parameters')
st.header('IRIS App')
Clf,Cls,Reg=st.tabs(['Classification','Cluster','Regression'])
def user_input():
    sepal_length=st.sidebar.slider('Sepal Length',4.3,7.9,5.8)
    sepal_width=st.sidebar.slider('Sepal Width',2.0,4.4,3.0)
    petal_length=st.sidebar.slider('Petal Length',1.0, 6.9, 4.35)
    petal_width=st.sidebar.slider('Petal Width',0.1, 2.5, 1.3)
    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width
          }
    df=pd.DataFrame(data,index=[0])
    return df



input=user_input()
st.write('User Input',input)
iris=load_iris()
X=iris.data
y=iris.target
button=st.sidebar.button('Predict')

with Clf:
    rfc=RandomForestClassifier(random_state=42)
    rfc.fit(X,y)


    if button:
        predict=rfc.predict(input)
        probability=rfc.predict_proba(input)
        st.write(f"Predicted Flower Species: {iris.target_names[predict][0]}")
        st.write('Predicted Class Index:',predict[0])

        st.subheader('Prediction Probability')
        st.write(probability)

with Cls:
    cluster=KMeans(n_clusters=3,random_state=42)
    cluster.fit(X)
    if button:
        predict=cluster.predict(input)
        st.write(f"The input data belongs to cluster: {predict[0]}")

        st.subheader('Cluster Centers')
        st.write(cluster.cluster_centers_)

        st.subheader('Inertia')
        st.write(cluster.inertia_)

with Reg:
    regression=RandomForestRegressor(n_estimators=100,random_state=42)
    X_reg=X[:, [0, 1, 3]]
    petal=X[:,2]
    regression.fit(X_reg,petal)
    if button:
        predict = regression.predict(input.iloc[:, [0, 1, 3]])
        st.write(f"Predicted Petal Length: {predict[0]:.2f} cm")

        st.subheader('Feature Importance')
        st.write(regression.feature_importances_)