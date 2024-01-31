from flask import Flask,redirect
#from flask import render_template
#from flask import request, url_for
import numpy as np
import pandas as pd
from flask import Flask,render_template,flash, redirect,url_for,session,logging,request
from flask_sqlalchemy import SQLAlchemy
from flask_toastr import Toastr



app = Flask(__name__, static_url_path='/imgs')

data=pd.read_csv('movie_metadata.csv')

data.shape

data['main_genre'] = data.genres.str.split('|').str[0]

data['content_rating'].value_counts()

#print(data.isna().sum())

d=data[['main_genre','actor_1_name','actor_2_name','director_name','language','imdb_score']]

data=d.dropna()

data.isna().sum()

#print(data['actor_2_name'].value_counts())


X=data[['main_genre','actor_1_name','actor_2_name','director_name','language']]
y=data['imdb_score']

bins=[1,5,10]
labels = ['FLOP', 'HIT']
y=pd.cut(y,bins=bins,labels=labels)

#print(y.value_counts())

from sklearn import preprocessing
l1 = preprocessing.LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()

ohe.fit(X)
X_1=ohe.transform(X).toarray()

y=l1.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()

sgd.fit(X_1,y)

sgd.score(X_test,y_test)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

y_pred=sgd.predict(X_test)


####                             ########             #########
import ast
from ast import literal_eval

d2 = pd.read_csv('demograph.csv', sep=',', encoding='latin-1')

d2=d2.drop(d2.filter(regex="Unnamed"),axis=1)

d2=d2.dropna()

d2.Genres=d2.Genres.apply(literal_eval)
d2['main_genre'] = d2['Genres'].str[0]

from sklearn.preprocessing import LabelEncoder
l2 = LabelEncoder()
d2['main_genre'] = l2.fit_transform(d2.main_genre)

d2['under18'] = d2.under18.str.split(',').str[1]
d2['18-29'] = d2.from18to29.str.split(',').str[1]
d2['30-44'] = d2.from30to44.str.split(',').str[1]
d2['45+'] = d2.age45.str.split(',').str[1]

d2["under18"] = d2["under18"].str.extract('(\d+)', expand=False).astype(int)

d2["18-29"] = d2["18-29"].str.extract('(\d+)', expand=False).astype(int)

d2["30-44"] = d2["30-44"].str.extract('(\d+)', expand=False).astype(int)

d2["45+"] = d2["45+"].str.extract('(\d+)', expand=False).astype(int)

d2.drop(columns=[ 'Movie', 'Genres', 'median', 'arithmetic_mean', 'number_of_votes',
       'under18', 'from18to29', 'from30to44', 'age45', 'Total_Male_data',
       'Total_Female_data', 'Male_under_18', 'Male_18_to_29', 'Male_30_to_44',
       'Male_45+', 'Female_under_18', 'Female_18_to_29', 'Female_30_to_44',
       'Female45+',])

from sklearn.preprocessing import OneHotEncoder, StandardScaler

d2['main_genre'] = pd.to_numeric(d2['main_genre'])
d2['under_18'] = pd.to_numeric(d2['under18'])
d2['18-29'] = pd.to_numeric(d2['18-29'])
d2['30-44'] = pd.to_numeric(d2['30-44'])
d2['45+'] = pd.to_numeric(d2['45+'])

# Prepare the input features and target variables
X2 = d2[["main_genre"]]
y_under_18 = d2["under18"]
y_18_29 = d2["18-29"]
y_30_44 = d2["30-44"]
y_45_plus = d2["45+"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
X_train2, X_test2, y_under_18_train, y_under_18_test, y_18_29_train, y_18_29_test, y_30_44_train, y_30_44_test, y_45_plus_train, y_45_plus_test = train_test_split(X2, y_under_18, y_18_29, y_30_44, y_45_plus, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled2 = scaler.fit_transform(X_train)
X_test_scaled2 = scaler.transform(X_test)

'''b = input('main_genre')
print(b)

b=np.array(b)

new_movie_genre_encoded = l2.transform(b.reshape(-1,1))

d2e=d2[d2['main_genre']==new_movie_genre_encoded[0]]

d2e['under18'].mean()

d2e['18-29'].mean()

d2e['30-44'].mean()

d2e['45+'].mean()'''

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/Aditya Bangali/Desktop/Project/database_new.db'
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

app.app_context().push()
toastr = Toastr(app)

toastr.init_app(app)


class user(db.Model):
    # __tablename__ = 'user'
    # __table_args__ = {'sqlite_autoincrement': True}
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password


@app.route('/',methods=['GET','POST'])
def mainLogin():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        
        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            # return redirect(url_for("error.html"))
            return render_template("index.html")
        else:
            #flash("Error!!!   Entered Username or password is wrong")
            return render_template("error.html")
    return render_template("mainLoginRegister_1.html")


@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        
        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            # return redirect(url_for("error.html"))
            return render_template("index.html")
        else:
            return render_template("error.html")
    return render_template("mainLoginRegister_1.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username = uname, email = mail, password = passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")


@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    if request.method=='POST':
        l=[]
        inp=request.form['inp'].split(',')
        inp=np.array(inp)
        t_inp=ohe.transform(inp.reshape(-1,5))
        ans=l1.inverse_transform((sgd.predict(t_inp)))[0]
        b=np.array(inp[0])
        new_movie_genre_encoded = l2.transform(b.reshape(-1,1))
        d2e=d2[d2['main_genre']==new_movie_genre_encoded[0]]
        l.extend([round(d2e['under18'].mean(),2),round(d2e['18-29'].mean(),2),round(d2e['30-44'].mean(),2),round(d2e['45+'].mean(),2)])
        return render_template('answer.html',l=l,ans=ans,b=b)
    



if __name__=='__main__':
    app.run()
