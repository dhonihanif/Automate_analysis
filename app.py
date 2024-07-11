import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

from lazypredict.Supervised import LazyClassifier, LazyRegressor

def data(data):
    # Load data
    df = pd.read_excel(data)
    for i in df.columns:
        if df[i].dtype == "object":
            df[i] = df[i].astype("category")
    
    return df

def split(df):
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def load_model(path):
    model = joblib.load(path)
    
    return model

def detect_outliers(df, name:str):
    try:
        Q1 = df[name].quantile(0.25)
        Q3 = df[name].quantile(0.75)
        IQR = Q3 - Q1
        
        min_IQR = Q1 - 1.5*IQR
        max_IQR = Q3 + 1.5*IQR
        outliers = df[(df[name] < min_IQR) | (df[name] > max_IQR)]
        return outliers
    except TypeError:
        raise TypeError("Type must be pandas dataframe")
    except:
        raise

def remove_outliers(data, name: str):
    try:
        Q1 = df[name].quantile(0.25)
        Q3 = df[name].quantile(0.75)
        IQR = Q3 - Q1
        
        min_IQR = Q1 - 1.5*IQR
        max_IQR = Q3 + 1.5*IQR
        outliers = df[(df[name] > min_IQR) | (df[name] > max_IQR)]
        return outliers
    except TypeError:
        raise TypeError("Type must be pandas dataframe")
    except:
        raise

option = st.sidebar.selectbox(
    "Silakan pilih:",
    ("Home", "Dataset", "Advanced Analysis", "Machine Learning")
)

if option == "Home" or option == "":
    st.write("""# Automate Analysis of Dataset""") #menampilkan halaman utama
    st.write()
    st.markdown("**This project shows the automate analysis of dataset**")
    st.write("This website is about build automate analysis of dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.image("./image/dhoni_photo.png", width=300)
    with col2:
        st.write(f"""
        Name : Dhoni Hanif Supriyadi\n
        Degree : Bachelor degree start from 2020 until 2024\n
        Lasted GPA : 3.98 from 4.00\n
        University : University of Bina Sarana Informatika\n
        Field : Information System\n
        """)

elif option == "Dataset":
    st.write("""## Dataset and Data Preparation""")
    st.write()
    uploaded_files = st.file_uploader("Choose a CSV file")
    df = ""
    if uploaded_files is not None:
        # To read file as bytes:
        bytes_data = uploaded_files.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_files)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        st.write(df)
        st.write(f"Total {df.shape[0]} row with {df.shape[1]} columns")
    a = st.button("Analysis Data for Cleaning", type="primary")
    if a and type(df) != "str":
        st.write("Analysis Data.....")
        st.write(f"Missing Values:\n {df.isnull().sum()}")
        st.write(f"Duplicate Values:\n {df.duplicated().sum()}")
        df.drop_duplicates(inplace=True)        
        categoric = [i for i in df.columns[:-1] if df[i].dtypes == "category" or df[i].dtypes == "object"]
        numeric = [i for i in df.columns[:-1] if i not in categoric]
        st.write("Outliers")
        results = []
        for i in numeric:
            outliers = detect_outliers(df, i)
            st.write(f"{i} : {outliers.shape[0]} outliers")
            if outliers.shape[0] <= (20/100) * df.shape[0]:
                df = remove_outliers(df, i)
            else:
                results.append(i)

        st.write("Cleaning Data....")
        for i in df.columns[:-1]:
            if df[i].isnull().sum() >= (60/100) * df.shape[0]:
                df.drop(i, axis=1, inplace=True)
        categoric = [i for i in df.columns[:-1] if df[i].dtypes == "category" or df[i].dtypes == "object"]
        numeric = [i for i in df.columns[:-1] if i not in categoric]
        df[categoric].fillna(df[categoric].mode(), axis=1, inplace=True)
        for i in numeric:
            if i not in results:
                df[i].fillna(df[i].mean(), inplace=True)
            else:
                df[i].fillna(df[i].median(), inplace=True)
        st.write(df)
        st.write(f"Total data has been cleaned {df.shape[0]} rows with {df.shape[1]} columns")

        csv = df.to_csv().encode("utf-8")
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="data.csv",
            mime="text/csv",
        )
   
    elif a == True and type(df) == "str":
        st.write("Input data first!")

elif option == "Advanced Analysis":
    st.write("""## Advance Analysis""")
    st.write()
    uploaded_files = st.file_uploader("Choose a CSV file")
    df = ""
    if uploaded_files is not None:
        # To read file as bytes:
        bytes_data = uploaded_files.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_files)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        st.write(df)
        st.write(f"Total {df.shape[0]} row with {df.shape[1]} columns")

        columns = df.columns
        opti = st.selectbox(
            "Univariate Analysis",
            ["Pick"] + [i for i in columns]
        )
        if opti != "Pick":
            fig, ax = plt.subplots()
            ax.hist(df[opti])
            ax.set_title(opti)
            st.pyplot(fig)
        
        opsi = st.selectbox(
            "Bivariate Analysis",
            ["Pick", "Analisis data numerik", "Analisis data kategori"]
        )
        columns_category = [i for i in df.columns if df[i].dtypes == "category" or df[i].dtypes == "object"]
        columns_numeric = [i for i in df.columns if i not in columns_category]
        if opsi == "Analisis data numerik":
            opsi1 = st.selectbox(
                "Bivariate1 Analysis",
                columns_numeric
            )
            opsi2 = st.selectbox(
                "Bivariate2 Analysis",
                columns_numeric
            )
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=opsi1, y=opsi2, ax=ax)
            ax.set_title(f"{opsi1} vs {opsi2}")
            st.pyplot(fig)
        elif opsi == "Analisis data kategori":
            opsi1 = st.selectbox(
                "Bivariate1 Analysis",
                columns_category
            )
            opsi2 = st.selectbox(
                "Bivariate2 Analysis",
                columns_numeric
            )
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=opsi1, y=opsi2, ax=ax)
            ax.set_title(f"{opsi1} vs {opsi2}")
            st.pyplot(fig)
        if df.shape[1] >= 3:
            ops = st.selectbox(
                "Multivariate Analysis",
                ["Pick", "Multivariate analysis 3 variable", "Multivariate analysis 4 variable"]
            )
            if ops == "Multivariate analysis 3 variable":
                ops1 = st.selectbox(
                    "Multivariate1 Analysis",
                    columns_category
                )
                ops2 = st.selectbox(
                    "Multivariate2 Analysis",
                    columns_numeric
                )
                ops3 = st.selectbox(
                    "Multivariate3 Analysis",
                    columns_numeric
                )
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=ops2, y=ops3, hue=ops1, ax=ax)
                ax.set_title(f"{ops1} vs {ops2} vs {ops3}")
                st.pyplot(fig)
            elif ops == "Multivariate analysis 4 variable":
                ops1 = st.selectbox(
                    "Multivariate1 Analysis",
                    columns_category
                )
                ops2 = st.selectbox(
                    "Multivariate2 Analysis",
                    columns_numeric
                )
                ops3 = st.selectbox(
                    "Multivariate3 Analysis",
                    columns_numeric
                )
                ops4 = st.selectbox(
                    "Multivariate4 Analysis",
                    columns_numeric
                )
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=ops2, y=ops3, hue=ops1, size=ops4, ax=ax)
                ax.set_title(f"{ops1} vs {ops2} vs {ops3} vs {ops4}")
                st.pyplot(fig)

elif option == "Machine Learning":
    st.write("""## Machine Learning""")
    st.write()
    uploaded_files = st.file_uploader("Choose a CSV file")
    if uploaded_files is not None:
        # To read file as bytes:
        bytes_data = uploaded_files.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_files.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_files)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        st.write(df)
        st.write(f"Total {df.shape[0]} row with {df.shape[1]} columns")
        columns = df.columns
        optio = st.selectbox(
            "Select Dependen Variable",
            ["Pick"] + [i for i in columns]
        )
        if optio != "Pick":
            X = df.drop(optio, axis=1)
            y = df[optio]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.write("Splitting Data...")
            st.write(f"Data after split into training and testing. Total for training {X_train.shape[0]} data and testing {X_test.shape[0]}")
            opt = st.selectbox(
                "Need Transformation or Not",
                ["Pick", "Minmaxscaler", "Standardscaler", "Robustscaler"]
            )
            categoric = [i for i in X.columns if X[i].dtypes == "object" or X[i].dtypes == "category"]
            numeric = [i for i in X.columns if i not in categoric]
            if opt == "Minmaxscaler":
                transform = MinMaxScaler().fit(X_train[numeric])
            elif opt == "Standardscaler":
                transform = StandardScaler.fit(X_train[numeric])
            elif opt == "Robustscaler":
                transform = RobustScaler().fit(X_train[numeric])
            categorical = {}
            if "transform" in vars():
                X_train[numeric] = transform.transform(X_train[numeric])
                X_test[numeric] = transform.transform(X_test[numeric])

                for i in categoric:
                    categorical[i] = LabelEncoder().fit(X_train[i])
                    X_train[i] = categorical[i].transform(X_train[i])
                    X_test[i] = categorical[i].transform(X_test[i])
                if y.dtypes == "object" or y.dtypes == "category":
                    label = LabelEncoder().fit(y)
                    y_train = label.transform(y_train)
                    y_test = label.transform(y_test)
                df_new = X_train.copy()
                df_new[optio] = y_train
                st.write(df_new)
                st.write("All data prepared!")
                task = st.selectbox(
                    "Pick Task of Business Problem",
                    ["Pick", "Classification task", "Regression task"]
                )
                ml = st.selectbox(
                    "Pick Machine Learning",
                    ["Pick", "Lazy Predict", "Linear/Logistic Regression", 
                        "K-Nearest Neighbors", "Decision Tree",
                        "Naive Bayes", "Support Vector Machine"]
                )
                if task == "Classification task":
                    if ml == "Linear/Logistic Regression":
                        model = LogisticRegression().fit(X_train, y_train)
                    elif ml == "K-Nearest Neighbors":
                        model = KNeighborsClassifier().fit(X_train, y_train)
                    elif ml == "Decision Tree":
                        model = DecisionTreeClassifier().fit(X_train, y_train)
                    elif ml == "Naive Bayes":
                        model = BernoulliNB().fit(X_train, y_train)
                    elif ml == "Support Vector Machine":
                        model = SVC().fit(X_train, y_train)
                    elif ml == "Lazy Predict":
                        clf = LazyClassifier(verbose=0, ignore_warnings=True)
                        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
                    
                    if "model" in vars():
                        y_pred = model.predict(X_test)
                        y_ = model.predict(X_train)
                        fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
                        sns.heatmap(confusion_matrix(y_train, y_), annot=True, fmt="d", linewidths=0.2, linecolor="white", cmap="viridis", ax=ax[0])
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", linewidths=0.2, linecolor="white", cmap="viridis", ax=ax[1])
                        ax[0].set_title("Training Data")
                        ax[1].set_title("Testing Data")
                        st.pyplot(fig)
                        st.write()
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average="micro")
                        recall = recall_score(y_test, y_pred, average="micro")
                        f1 = f1_score(y_test, y_pred, average="micro")
                        evaluates = pd.DataFrame({"Accuracy Score": f"{accuracy*100:.2f}%",
                                                  "Precision Score": f"{precision*100:.2f}%",
                                                  "Recall Score": f"{recall*100:.2f}%",
                                                  "F1 Score": f"{f1*100:.2f}%"}, index=[0, 1, 2, 3])
                        
                    elif "models" in vars():
                        st.write(models)

                elif task == "Regression task":
                    if ml == "Linear/Logistic Regression":
                        model = LinearRegression().fit(X_train, y_train)
                    elif ml == "K-Nearest Neighbors":
                        model = KNeighborsRegressor().fit(X_train, y_train)
                    elif ml == "Decision Tree":
                        model = DecisionTreeRegressor().fit(X_train, y_train)
                    elif ml == "Naive Bayes":
                        model = GaussianNB().fit(X_train, y_train)
                    elif ml == "Support Vector Machine":
                        model = SVR().fit(X_train, y_train)
                    elif ml == "Lazy Predict":
                        reg = LazyRegressor(verbose=0, ignore_warnings=True)
                        models, predictions = reg.fit(X_train, X_test, y_train, y_test)

                    if "model" in vars():
                        y_pred = model.predict(X_test)
                        y_ = model.predict(X_train)
                        fig, ax = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
                        sns.scatterplot(x=X_train[numeric[0]], y=y_train, ax=ax[0])
                        sns.scatterplot(x=X_test[numeric[0]], y=y_test, ax=ax[1])
                        sns.lineplot(x=X_train[numeric[0]], y=y_, ax=ax[0])
                        sns.lineplot(x=X_test[numeric[0]], y=y_test, ax=ax[1])

                        ax[0].set_title("Training Predict")
                        ax[1].set_title("Testing Predict")
                        st.pyplot(fig)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = mse ** (1/2)
                        mae = mean_absolute_error(y_test, y_pred)
                        evaluates = pd.DataFrame({"R2 Score": f"{r2*100:.2f}%",
                                                  "Mean Squared Error": f"{mse*100:.2f}%",
                                                  "Root Mean Squared Error": f"{rmse*100:.2f}%",
                                                  "Mean Absolute Error": f"{mae*100:.2f}%"}, index=[0, 1, 2, 3])
                        st.write(evaluates)
                    elif "models" in vars():
                        st.write(models)
