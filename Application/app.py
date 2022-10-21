#Using flask to create a Web App
import scipy.sparse
from flask import Flask, render_template, redirect, make_response
#Creating the database id and password for a secure network
from flask_sqlalchemy import SQLAlchemy

#Using json to mediate the graphs from plotly to flask
import json
import plotly.utils

#Libraries used for machine learning and graphing
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.linear_model import LogisticRegression




app = Flask(__name__, template_folder= "template")



#Creating a function to clean the data
def data_cleaning(data):
    # Remove NA values
    data = data.dropna()

    # Replace 'Type of Trip:' to numbers to represent each type of the trip
    data = data.replace(to_replace="Undisclosed",
                        value="0")
    data = data.replace(to_replace="Travelled solo",
                        value="1")
    data = data.replace(to_replace="Travelled as a couple",
                        value="2")
    data = data.replace(to_replace="Travelled with friends",
                        value="3")
    data = data.replace(to_replace="Travelled with family",
                        value="4")
    data: object = data.replace(to_replace="Travelled on business",
                                value="5")

    # Changing "Month of Stay:" into 2 columns, 1 for Year and 1 for Month
    # data = data['Month']
    data['Month & Year'] = data['Month of Stay:']
    data['Month & Year'] = pd.to_datetime(data['Month & Year'], format='%b-%y')
    data[['Month', 'Year']] = data['Month of Stay:'].str.split("-", expand=True, )

    def Month(x):
        if x == 'Jan':
            return (int(1))
        if x == 'Feb':
            return (int(2))
        if x == 'Mar':
            return (int(3))
        if x == 'Apr':
            return (int(4))
        if x == 'May':
            return (int(5))
        if x == 'Jun':
            return (int(6))
        if x == 'Jul':
            return (int(7))
        if x == 'Aug':
            return (int(8))
        if x == 'Sep':
            return (int(9))
        if x == 'Oct':
            return (int(10))
        if x == 'Nov':
            return (int(11))
        if x == 'Dec':
            return (int(12))

    data['Month'] = data['Month'].apply(Month)
    data = data.drop('Month of Stay:', axis=1)

    # Changing data type to int
    data['Type of Trip:'] = data['Type of Trip:'].astype('int')
    return (data)

#Letting the system recognise the cleaned data
with app.app_context():
    Yotel_data = "./static/data/Yotel.csv"
    Yoteldata = pd.read_csv(Yotel_data)

    HotelG_data = "./static/data/HotelG.csv"
    HotelGdata = pd.read_csv(HotelG_data)

    ShangriLa_data = "./static/data/ShangriLa.csv"
    ShangriLadata = pd.read_csv(ShangriLa_data)

    Fullerton_data = "./static/data/Fullerton.csv"
    Fullertondata = pd.read_csv(Fullerton_data)

    Carlton_data = "./static/data/carlton.csv"
    Carltondata = pd.read_csv(Carlton_data)

    Overall_data = "./static/data/Combined.csv"
    Overalldata = pd.read_csv(Overall_data)

    Yotelcdata = data_cleaning(Yoteldata)
    HotelGcdata = data_cleaning(HotelGdata)
    ShangriLacdata = data_cleaning(ShangriLadata)
    Fullertoncdata = data_cleaning(Fullertondata)
    Carltoncdata = data_cleaning(Carltondata)
    Overallcdata = data_cleaning(Overalldata)


def find_vif(xdata):
    naming1 = str(xdata.iat[1, 6])
    vifdata = xdata[
        ["Review Rating", "Type of Trip:", "No. of contributions from reviewer:", "No. of helpful votes on review:",
         "Month"]]

    # Randomising data
    vif_data = vifdata.sample(n=len(vifdata))
    vif_data = pd.DataFrame()

    vif_data["Features"] = vifdata.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(vifdata.values, i)
                       for i in range(len(vifdata.columns))]

    if naming1 == 'Carlton':
        check1 = str(xdata.iat[-1, 6])
        if check1 == naming1    :
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(vif_data.columns),
                            fill_color='orange',
                            font=dict(color='white', size=14),
                            align='center'),
                cells=dict(values=[vif_data.Features, vif_data.VIF],
                           fill_color='moccasin',
                           align=['center', 'center']))
            ])
            fig.update_layout(title="VIF value for " + str(naming1))
            vif = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return vif
        else:
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(vif_data.columns),
                            fill_color='orange',
                            font=dict(color='white', size=14),
                            align='center'),
                cells=dict(values=[vif_data.Features, vif_data.VIF],
                           fill_color='moccasin',
                           align=['center', 'center']))
            ])
            fig.update_layout(title="VIF value for all hotels")
            vif = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return vif

    else:
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(vif_data.columns),
                        fill_color='orange',
                        font=dict(color='white', size=14),
                        align='center'),
            cells=dict(values=[vif_data.Features, vif_data.VIF],
                       fill_color='moccasin',
                       align=['center', 'center']))
        ])
        fig.update_layout(title="VIF value for " + str(naming1))
        vif = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return vif



def Positiverating(x):
    if x >= 30:
        return (int(1))
    else:
        return (int(0))

def ROC_Curve_with_AUC_RMSE(Modeldata):
    Modeldata['Positiverating'] = Modeldata['Review Rating'].apply(Positiverating)
    Modeldata.sample(n=len(Modeldata))
    naming = str(Modeldata.iat[1, 6])
    check = str(Modeldata.iat[-1,6])

    X = Modeldata[['Type of Trip:', 'No. of contributions from reviewer:', 'No. of helpful votes on review:', 'Month']]
    y = Modeldata['Positiverating']

    model = LogisticRegression()
    model.fit(X, y)
    y_score = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_score)
    RMSE = np.sqrt(mean_squared_error(y, y_score))
    RMSE = np.sqrt(mean_squared_error(y,y_score))
    RMSE = str(round(RMSE,4))
    AUC = auc(fpr,tpr)
    AUC = str(round(AUC,4))


    if naming == "Yotel":
        if naming == check:
            fig = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve for '+naming+' (AUC = '+AUC+' and RMSE = ' + RMSE + ')',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return graphJSON

        else:
            fig = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve for '+naming+' (AUC = '+AUC+' and RMSE = ' + RMSE + ')',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return graphJSON

    else:
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve for ' + naming + ' (AUC = ' + AUC + ' and RMSE = ' + RMSE + ')',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON

#General Charts
def percentage_pie(datax):
    labels = datax['Type of Trip:']
    values = datax
    df = datax
    print(df)
    naming1 = str(datax.iat[1, 7])
    print(naming1)
    check = str(datax.iat[-1, 7])
    df = df.groupby(['Type of Trip:'], as_index=False).count()
    df = df.loc[df['Type of Trip:'] != 'Undisclosed']
    if naming1 =="Yotel":
        if check == "Yotel":
            fig = px.pie(df, values='Review Title', names='Type of Trip:',
                         title="Percentage on the Type of Trip booked by customers for "+naming1)
            pie = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return pie
        else:
            fig = px.pie(df, values='Review Title', names='Type of Trip:',
                         title="Percentage on the Type of Trip booked by customers overall")
            pie = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return pie
    else:
        fig = px.pie(df, values='Review Title', names='Type of Trip:', title="Percentage on the Type of Trip booked by customers for "+naming1)
        pie = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return pie

def no_of_cust_over_year(df1):
    df2 = df1.loc[df1.Year.isin(['22', '21'])]
    df2 = df2.groupby(['Month & Year','Hotels'], as_index=False).count()

    fig = px.line(df2, x="Month & Year", y="Review Rating", color='Hotels', labels={"Review Rating": "Number of customers"}, title="Number of customers that stayed in the hotel by Month")
    cust_over_year = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return cust_over_year


def bar_review_rating(datax):
    graph = datax.groupby(['Hotels','Review Rating'], as_index=False).count()
    fig = px.bar(graph, x="Review Rating", y="Review Title", color="Hotels", title="Number of Review Rating by hotels", text_auto=True)
    Bar_review = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return Bar_review

 # Creating a table graph to showcase abbreviation and its description
def vif_table(datax):
    values = [['<b>VIF<br>(Variance Inflation<br>Factor)</b>', '<b>RMSE<br>(Root Mean <br>Square Error)</b>',
               '<b>ROC Curve<br>(Receiver Operating <br>Characteristic Curve)</b>',
               '<b>AUC<br>(Area Under the<br>ROC Curve)</b>'],  # 1st col
              [
                  "A measure to detect any multicollinearity and its severity between independent variables. It is essential to compute VIF to ensure there are no inputs that are influencing each other, which might potentially cause the prediction model to be inaccurate. Multicollinearity is present when any of the value is more than 10.",
                  "A measure to compute the standard deviation and the spread of the data points from the regression line. A lower RMSE value (usually less than 1) is considered a good indicator.",
                  "A graph to illustrate the performance of the predictive model in classifying at different thresholds. The graph is plotted with two parameters, 'True Positive Rate' and 'False Positive Rate'. The higher the arc of the curve, the better the model is at classifying true positive.",
                  "A measure to compute the area under the curve created by the ROC Curve. AUC ranges from 0 to 1, where 1 indicates that the model is able to predict 100% correctly and 0 indicates 100% wrong prediction. An acceptable value of AUC is more than 0.5."]]

    Table = go.Figure(data=[go.Table(
        columnorder=[1, 2],
        columnwidth=[95, 400],
        header=dict(
            values=[['<b>ABBREVIATION</b>'],
                    ['<b>DESCRIPTION</b>']],
            line_color='darkslategray',
            fill_color='orange',
            align=['center', 'center'],
            font=dict(color='white', size=13),
            height=40
        ),
        cells=dict(
            values=values,
            line_color='darkslategray',
            fill=dict(color=['moccasin', 'white']),
            align=['left', 'left'],
            font_size=12,
            height=30)
    )
    ])
    Show_table = json.dumps(Table, cls=plotly.utils.PlotlyJSONEncoder)
    return Show_table




#ID and directory for SQLAlchemy (Database)
app.config['SECRET_KEY'] = "HBBjdnklmdkmdmkkJIJIUOWHUdihiuwiu211992389!"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///CustomerProfile.db'
db = SQLAlchemy(app)


#Inputs of the database
class CustomerProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Hotel = db.Column(db.String(50), default='Fullerton', nullable=False)
    Type = db.Column(db.String(50), nullable=False, default='Travelled solo')
    Month = db.Column(db.String(50), default='January', nullable=False)
    HelpfulVotes = db.Column(db.Integer, nullable=False)
    Contributions = db.Column(db.Integer, nullable=False)
    Review = db.Column(db.String(5000), nullable=False)


#Creating of the input forms
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField, StringField
from wtforms.validators import DataRequired

class UserDataForm(FlaskForm):
    #Creating inputs for the users to choose
    Hotel = SelectField('Hotel', validators=[DataRequired()],
                                choices=[('Fullerton', 'Fullerton'),
                                        ('Carlton', 'Carlton'),
                                         ('HotelG', 'HotelG'),
                                          ('ShangriLa', 'ShangriLa'),
                                           ('Yotel', 'Yotel')])
    Type = SelectField("Type of Trip", validators=[DataRequired()],
                                            choices =[('Travelled solo', 'Travelled solo'),
                                            ('Travelled with family', 'Travelled with family'),
                                            ('Travelled on business', 'Travelled on business'),
                                            ('Travelled with friends', 'Travelled with friends'),
                                            ('Travelled as a couple','Travelled as a couple')])
    Month = SelectField("Month of stay", validators=[DataRequired()], choices = [('January','January'),
                                                                                 ('February','February'),
                                                                                 ('March','March'),
                                                                                 ('April','April'),
                                                                                 ('May', 'May'),
                                                                                 ('June', 'June'),
                                                                                 ('July', 'July'),
                                                                                 ('August', 'August'),
                                                                                 ('September', 'September'),
                                                                                 ('October', 'October'),
                                                                                 ('November', 'November'),
                                                                                 ('December', 'December')])
    HelpfulVotes = IntegerField('No. of helpful votes on review', validators=[DataRequired()])
    Contributions = IntegerField('No. of contributions from reviewer', validators = [DataRequired()])
    Review = StringField("Review description", validators=[DataRequired()])
    submit = SubmitField('Submit your customer profile')

# Creating the database
with app.app_context():
    db.create_all()
    DescriptionList = []


@app.route('/add', methods = ["GET","POST"])
def add():

    form = UserDataForm()
    entries = CustomerProfile.query
    #Creating a dummy entry for database
    entry1 = CustomerProfile(Hotel="Fullerton", Type="Travelled solo", Month="January", HelpfulVotes="20",
                             Contributions="10", Review="This is decent")
    #If form is valid, then submit the form into the database
    if form.validate_on_submit():
        entry = CustomerProfile(Hotel=form.Hotel.data, Type=form.Type.data, Month=form.Month.data, HelpfulVotes=form.HelpfulVotes.data, Contributions=form.Contributions.data, Review=form.Review.data)
        db.session.add(entry)
        db.session.commit()
        DescriptionList.append(form.Review.data)



        return redirect('/add')
    return render_template('add.html', title="Add expenses", form=form, entries=entries)

#Creating a button to delete the entry number
@app.route('/delete-post/<int:entry_id>')
def delete(entry_id):
    entry = CustomerProfile.query.get_or_404(int(entry_id))
    db.session.delete(entry)
    db.session.commit()
    # useridList.pop(int(entry_id) - 1)
    return redirect('/add')







#Creating a route for the home page
@app.route("/", methods=["GET","POST"])
def home():
    return render_template("home.html")



#Creating a route to comparison whereby hotels are able to compare Review Ratings against other variables
@app.route("/comparison", methods=["GET","POST"])
def comparison():
    fig1 = no_of_cust_over_year(Overallcdata)


    #Creating the graph between type of trips and Review rating
    Overall = "./static/data/Combined.csv"
    df = pd.read_csv(Overall)

    dfg = df.groupby(['Type of Trip:','Hotels'], as_index=False).count()
    dfg= dfg.loc[dfg['Type of Trip:'] != "Undisclosed"]

    bar = px.bar(dfg ,x="Type of Trip:", y="Review Rating", barmode= "group", color="Hotels", labels={"Review Rating":"No. of customers","Type of Trip:":"Type of Trip"})

    bar.update_traces(dict(marker_line_width=0))
    fig2 = json.dumps(bar, cls=plotly.utils.PlotlyJSONEncoder)

    fig3=bar_review_rating(Overallcdata)

    fig4 = ROC_Curve_with_AUC_RMSE(Overallcdata)

    fig5= find_vif(Overallcdata)

    fig6=vif_table(Overallcdata)



    return render_template("comparison.html", OverallGraph1=fig1, OverallGraph2=fig2, OverallGraph3=fig3, OverallGraph4=fig4, OverallGraph5=fig5, OverallGraph6=fig6    )

#Creating a route for fullerton
@app.route("/Fullerton")
def FullertonMain():
    # Using sentiment analysis to chart graphs
    FullertonSA = "./static/data/Fullerton_SA.csv"
    df = pd.read_csv(FullertonSA)
    positive = df[df["Sentiment"] == 2]
    neutral = df[df["Sentiment"] == 1]
    negative = df[df["Sentiment"] == 0]
    pos = len(positive) * 100 / len(df['Sentiment'])
    neu = len(neutral) * 100 / len(df['Sentiment'])
    neg = len(negative) * 100 / len(df['Sentiment'])

    percentage = [neu, neg, pos]
    labels = ["Neutral", "Negative", "Positive"]

    fig3 = px.pie(values=percentage, names=labels, color=labels, color_discrete_map={'Positive': 'green',
                                                                                     'Neutral': 'blue',
                                                                                     'Negative': 'tomato'})
    fig3.update_layout(title="Percentage of the reviews for Fullerton Hotel")
    fig3.update_traces(textposition='inside',
                       textinfo='percent', showlegend=True)
    Fullerton1 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    fig4 = px.histogram(Fullertondata, x="Review Rating", title='Review Rating for Fullerton Hotel')
    fig4.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                       marker_line_width=1.5)
    Fullerton2 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    Fullerton3 = percentage_pie(Fullertondata)

    Fullerton4 = find_vif(Fullertoncdata)

    Fullerton5= vif_table(Fullertoncdata)

    Fullerton6= ROC_Curve_with_AUC_RMSE(Fullertoncdata)


    return render_template("FullertonMain.html", HotelGraph1=Fullerton1, HotelGraph2=Fullerton2, HotelGraph3=Fullerton5,HotelGraph4=Fullerton4, HotelGraph5=Fullerton3, HotelGraph6=Fullerton6)

@app.route("/HotelG")
def HotelGMain():
    HotelGSA = "./static/data/HotelG_SA.csv"
    df = pd.read_csv(HotelGSA)
    positive = df[df["Sentiment"] == 2]
    neutral = df[df["Sentiment"] == 1]
    negative = df[df["Sentiment"] == 0]
    pos = len(positive) * 100 / len(df['Sentiment'])
    neu = len(neutral) * 100 / len(df['Sentiment'])
    neg = len(negative) * 100 / len(df['Sentiment'])

    percentage = [neu, neg, pos]
    labels = ["Neutral", "Negative", "Positive"]

    fig1 = px.pie(values=percentage, names=labels, color=labels, color_discrete_map={'Positive': 'green',
                                                                                     'Neutral': 'blue',
                                                                                     'Negative': 'tomato'})
    fig1.update_layout(title="Percentage of the reviews for HotelG")
    fig1.update_traces(textposition='inside',
                       textinfo='percent', showlegend=True)
    HotelG1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.histogram(HotelGdata, x="Review Rating", title='Review Rating for HotelG')
    fig2.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                       marker_line_width=1.5)
    HotelG2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    HotelG3= percentage_pie(HotelGdata)

    HotelG4= find_vif(HotelGcdata)

    HotelG5= vif_table(HotelGdata)

    HotelG6 = ROC_Curve_with_AUC_RMSE(HotelGcdata)

    return render_template("HotelGMain.html", HotelGraph1=HotelG1, HotelGraph2=HotelG2, HotelGraph3=HotelG5, HotelGraph4=HotelG4, HotelGraph5=HotelG3, HotelGraph6=HotelG6)

@app.route("/ShangriLa")
def ShangriLaMain():
    ShangriLaSA = "./static/data/ShangriLa_SA.csv"
    df = pd.read_csv(ShangriLaSA)
    positive = df[df["Sentiment"] == 2]
    neutral = df[df["Sentiment"] == 1]
    negative = df[df["Sentiment"] == 0]
    pos = len(positive) * 100 / len(df['Sentiment'])
    neu = len(neutral) * 100 / len(df['Sentiment'])
    neg = len(negative) * 100 / len(df['Sentiment'])

    percentage = [neu, neg, pos]
    labels = ["Neutral", "Negative", "Positive"]

    fig1 = px.pie(values=percentage, names=labels, color=labels, color_discrete_map={'Positive': 'green',
                                                                                     'Neutral': 'blue',
                                                                                     'Negative': 'tomato'})
    fig1.update_layout(title="Percentage of the reviews for ShangriLa")
    fig1.update_traces(textposition='inside',
                       textinfo='percent', showlegend=True)
    ShangriLa1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.histogram(ShangriLadata, x="Review Rating", title='Review Rating')
    fig2.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                       marker_line_width=1.5)
    ShangriLa2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    ShangriLa3= percentage_pie(ShangriLadata)

    ShangriLa4= find_vif(ShangriLacdata)

    ShangriLa5 = vif_table(ShangriLacdata)

    ShangriLa6 = ROC_Curve_with_AUC_RMSE(ShangriLacdata)
    return render_template("ShangriLaMain.html", HotelGraph1=ShangriLa1, HotelGraph2=ShangriLa2, HotelGraph3=ShangriLa5, HotelGraph4=ShangriLa4, HotelGraph5=ShangriLa3, HotelGraph6=ShangriLa6)


@app.route("/Yotel")
def YotelMain():
    YotelSA = "./static/data/Yotel_SA.csv"
    df = pd.read_csv(YotelSA)
    positive = df[df["Sentiment"] == 2]
    neutral = df[df["Sentiment"] == 1]
    negative = df[df["Sentiment"] == 0]
    pos = len(positive) * 100 / len(df['Sentiment'])
    neu = len(neutral) * 100 / len(df['Sentiment'])
    neg = len(negative) * 100 / len(df['Sentiment'])

    percentage = [neu, neg, pos]
    labels = ["Neutral", "Negative", "Positive"]

    fig1 = px.pie(values=percentage, names=labels, color=labels, color_discrete_map={'Positive': 'green',
                                                                                     'Neutral': 'blue',
                                                                                     'Negative': 'tomato'})
    fig1.update_layout(title="Percentage of the reviews for Yotel")
    fig1.update_traces(textposition='inside',
                       textinfo='percent', showlegend=True)
    Yotel1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.histogram(Yoteldata, x="Review Rating", title='Review Rating')
    fig2.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                       marker_line_width=1.5)
    Yotel2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    Yotel3 = percentage_pie(Yoteldata)

    Yotel4= find_vif(Yotelcdata)

    Yotel5= vif_table(Yotelcdata)

    Yotel6=ROC_Curve_with_AUC_RMSE(Yotelcdata)

    return render_template("YotelMain.html", HotelGraph1=Yotel1, HotelGraph2=Yotel2, HotelGraph3=Yotel5, HotelGraph4=Yotel4, HotelGraph5= Yotel3, HotelGraph6=Yotel6)

@app.route('/Carlton')
def Carlton():
    CarltonSA = "./static/data/Carlton_SA.csv"
    df = pd.read_csv(CarltonSA)
    positive = df[df["Sentiment"] == 2]
    neutral = df[df["Sentiment"] == 1]
    negative = df[df["Sentiment"] == 0]
    pos = len(positive) * 100 / len(df['Sentiment'])
    neu = len(neutral) * 100 / len(df['Sentiment'])
    neg = len(negative) * 100 / len(df['Sentiment'])

    percentage = [neu, neg, pos]
    labels = ["Neutral", "Negative", "Positive"]

    pie_chart = px.pie(values=percentage, names=labels, color=labels, color_discrete_map={'Positive': 'green',
                                                                                     'Neutral': 'blue',
                                                                                     'Negative': 'tomato'})
    pie_chart.update_layout(title="Percentage of the reviews for Carlton")
    pie_chart.update_traces(textposition='inside',
                       textinfo='percent', showlegend=True)
    fig1 = json.dumps(pie_chart, cls=plotly.utils.PlotlyJSONEncoder)

    histogram = px.histogram(Carltondata, x="Review Rating", title='Review Rating')
    histogram.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                       marker_line_width=1.5)
    fig2 = json.dumps(histogram, cls=plotly.utils.PlotlyJSONEncoder)

    fig3= percentage_pie(Carltondata)

    fig4= find_vif(Carltoncdata)
    fig5=vif_table(Carltoncdata)
    fig6=ROC_Curve_with_AUC_RMSE(Carltoncdata)
    return render_template("CarltonMain.html", HotelGraph1=fig1, HotelGraph2=fig2, HotelGraph3=fig5, HotelGraph4=fig4, HotelGraph5=fig3, HotelGraph6=fig6)


@app.route('/downloadCarlton')
def downloadCarlton():
    path = './static/data/Carlton.csv '
    df=pd.read_csv(path)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Carlton excel sheet.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/downloadFullerton')
def downloadFullerton():
    path = './static/data/Fullerton.csv '
    df = pd.read_csv(path)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Fullerton excel sheet.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route('/downloadHotelG')
def downloadHotelG():
    path = './static/data/HotelG.csv '
    df = pd.read_csv(path)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Hotel G excel sheet.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/downloadShangriLa')
def downloadShangriLa():
    path = './static/data/ShangriLa.csv '
    df = pd.read_csv(path)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=ShangriLa excel sheet.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/downloadYotel')
def downloadYotel():
    path = './static/data/Yotel.csv '
    df = pd.read_csv(path)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=Yotel excel sheet.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/Chatbot')
def chatbot():
    return render_template("index.html")


#Function to run the app
if __name__ == "__main__" :
    app.run(debug= True)
# !/usr/bin/env python
    # coding: utf-8

#     # Import the Required Packages
#     from tensorflow import keras
#
#     print('keras: ', keras.__version__)
#     from IPython.core.display import display, HTML
#
#     display(HTML("<style>.container { width:95% !important; }</style>"))
#     # Import the Required Packages
#     import numpy as np
#     import pandas as pd
#     import os
#     import matplotlib.pyplot as plt
#     import tensorflow as tf
#     import nltk
#     from nltk.corpus import stopwords
#     from nltk.tokenize import word_tokenize
#     from tensorflow.python import keras
#
#     from keras.utils import to_categorical
#
#     from sklearn.model_selection import train_test_split
#     from keras.callbacks import EarlyStopping
#
#     from keras import optimizers
#
#     from keras.models import Sequential
#
#     from keras.layers import Dense, Embedding, Dropout
#
#     from keras.layers import LSTM,SimpleRNN, GRU, Bidirectional
#     # from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, LSTM!!, Bidirectional
#
#     from keras import layers,callbacks,regularizers
#
#
#     from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#     import seaborn as sns
#     import plotly.express as px
#     import re
#
#     from keras import backend as K
#
#     from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
#     from keras.preprocessing.text import Tokenizer
#
#     from keras_preprocessing.sequence import pad_sequences
#
#
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     from wordcloud import WordCloud
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import accuracy_score
#
#     from scipy import sparse
#     from sklearn.feature_extraction.text import TfidfVectorizer
#
#     # In[121]:
#
#
#
#
# @app.route('/prediction/<int:entry_id>')
# def show_description(entry_id):
#     test = {}
#     # This prints out a specific entry review under description
#     test[entry_id] = DescriptionList[entry_id - 1]
#     description = []
#     for i in test.values():
#         description.append(i)
#     # reading of csv
#     df = pd.read_csv('./static/data/Fullerton.csv', delimiter=',')
#
#     # making ratings be from 1-5 instead
#     df["Review Rating"] = df["Review Rating"].replace(10, 1)
#     df["Review Rating"] = df["Review Rating"].replace(20, 2)
#     df["Review Rating"] = df["Review Rating"].replace(30, 3)
#     df["Review Rating"] = df["Review Rating"].replace(40, 4)
#     df["Review Rating"] = df["Review Rating"].replace(50, 5)
#
#     df['Sentiment'] = np.select([df['Review Rating'] > 3, df['Review Rating'] == 3], [2, 1], 0)
#     # if at least 4/5 -> positive Sentiment, 3 neutral, 1 or 2 -> negative
#
#     # data preprocessing
#     # check if text is in English
#     import re
#
#     def isEnglish(text):
#         try:
#             text.encode(encoding='utf-8').decode('ascii')
#         except UnicodeDecodeError:
#             text = " "
#         else:
#             text = text
#         return text
#
#     def pre_process(text):
#         if type(text) == float:
#             return ""
#
#             # Convert to string
#         text = str(text)
#
#         # Convert to lowercase
#         text = text.lower()
#
#         # remove punctuations
#         text = re.sub("[^\w\s]", " ", text)
#
#         # remove tags
#         text = re.sub("</?.*?>", " <> ", text)
#
#         # remove special characters and digits
#         text = re.sub("(\\d|\\W|_)+", " ", text)
#
#         text = re.sub("'", "", text)  # This is to avoid removing contractions in english
#         text = re.sub("@[A-Za-z0-9_]+", "", text)
#         text = re.sub("#[A-Za-z0-9_]+", "", text)
#         text = re.sub(r'http\S+', '', text)
#         text = re.sub('[()!?]', ' ', text)
#         text = re.sub('\[.*?\]', ' ', text)
#         text = re.sub("[^a-z0-9]", " ", text)
#         text = text.split()
#         stop_words = set(stopwords.words('english'))
#         text = [w for w in text if not w in stop_words]
#         text = " ".join(word for word in text)
#         return text
#
#     # remove all blanks
#     df.replace(' ', np.nan, inplace=True)
#     df.dropna(subset=["Review Description"], inplace=True)
#
#     # Remove everything
#     df['Review Description'] = df['Review Description'].apply(lambda x: pre_process(x))
#
#     # Filter out words that is not english
#     df['Review Description'] = df['Review Description'].apply(lambda x: isEnglish(x))
#
#     # remove all blanks
#     df.replace(' ', np.nan, inplace=True)
#     df.dropna(subset=["Review Title"], inplace=True)
#
#     # Remove everything
#     df['Review Title'] = df['Review Title'].apply(lambda x: pre_process(x))
#
#     # Filter out words that is not english
#     df['Review Title'] = df['Review Title'].apply(lambda x: isEnglish(x))
#
#     # ## train model to get sentiment analysis
#
#     df["Review"] = df["Review Title"] + df["Review Description"]
#
#     df_new = df[['Review', 'Sentiment']]
#
#     fullreview = df_new['Review'].values
#
#     # In[39]:
#
#     sent = df_new.loc[:, 'Sentiment'].values
#
#     # In[40]:
#
#     max_len = 100  # We will cut reviews after 100 words
#     max_words = 50000  # We will only consider the top 50,000 words (vocabulary) in the dataset (dictionary)
#     # tokenizer
#     tokenizer = Tokenizer(num_words=max_words)
#     tokenizer.fit_on_texts(fullreview)
#     encoded_rev = tokenizer.texts_to_sequences(fullreview)
#     padded_sequence = pad_sequences(encoded_rev, maxlen=max_len)
#
#     word_index = tokenizer.word_index
#     print('Found %s unique tokens.' % len(word_index))
#
#     x = padded_sequence
#
#     y = to_categorical(np.asarray(sent))
#     print('Shape of data tensor:', x.shape)
#     print('Shape of label tensor:', y.shape)
#
#     # In[41]:
#
#     # splitting train and test data into 80% and 20%
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)
#
#     # ## GRU
#
#     model = Sequential()
#     model.add(Embedding(max_words, 32, input_length=max_len))
#     model.add(GRU(16, return_sequences=True))
#     model.add(Dropout(0.5))
#     model.add(GRU(16))
#     model.add(Dense(3, activation='softmax'))
#
#     model.summary()
#
#     model.compile(optimizer=optimizers.RMSprop(),
#                   loss='categorical_crossentropy',
#                   metrics=['acc'])
#
#     history = model.fit(X_train, y_train,
#                         epochs=12,
#                         batch_size=32,
#                         validation_split=0.2)
#
#
#     text_input = np.array(description)
#
#     sequences = tokenizer.texts_to_sequences(text_input)
#
#     X = pad_sequences(sequences, maxlen=max_len)
#
#     y = np.asarray(sent)
#
#     # Load the ratings_dictionary
#     df_map = pd.read_csv('./static/data/mapping.csv', delimiter=',')
#     sent_dictionary = df_map.loc[:, 'sentiment'].to_dict()
#
#     def prediction(model, text_input, items_l):
#         sequences = tokenizer.texts_to_sequences(text_input)
#         data = pad_sequences(sequences, maxlen=max_len)
#         prob = model.predict(data)
#         pro_df = pd.DataFrame(prob, columns=items_l)
#         result = items_l[np.argmax(prob)]
#         return pro_df, result
#     # OUTPUT
#     # print('User input: ', text_input)
#     def OUTPUT(text_input):
#         prob_df, result = prediction(model, text_input, np.unique(sent))
#         return (result, '\n\n', prob_df.to_string(index=False))
#
#     return OUTPUT(description)

    # ## Logistic Regression and Neural Network

    # train, val = train_test_split(df_new, test_size=0.2, random_state=5, stratify=df_new["Sentiment"])
    #
    # # to count the length of a review
    # def leng(col):
    #     l = []
    #     for rew in col:
    #         text = word_tokenize(rew, language='english')
    #         nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    #         filtered = [w for w in text if nonPunct.match(w)]
    #         l.append(len(filtered))
    #     return l
    #
    # # In[158]:
    #
    # train['wordCount'] = leng(train['Review'])
    #
    # # calculate mean and std to rescale the word count to be centered around 0
    # wrdcm = train['wordCount'].mean()
    # wrdcs = train['wordCount'].std()
    #
    # train['scaledwordCount'] = (train['wordCount'] - wrdcm) / wrdcs
    #
    # val['wordCount'] = leng(val['Review'])
    #
    # val['scaledwordCount'] = (val['wordCount'] - wrdcm) / wrdcs
    #
    # # In[159]:
    #
    # stpwrdlist = set(stopwords.words('english'))
    # vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.005, stop_words=stpwrdlist, ngram_range=(1, 3))
    #
    # # In[160]:
    #
    # # fit and transform the reviews
    #
    # X = vectorizer.fit_transform(train['Review'])
    # X_val = vectorizer.transform(val['Review'])
    #
    # # In[162]:
    #
    # lr = LogisticRegression(multi_class="multinomial")
    #
    # labels = ["negative", "neutral", "positive"]
    #
    # # In[163]:
    #
    # X = ((X, np.array(train['scaledwordCount'])[:, None]))
    #
    # X_val = scipy.sparse.hstack((X_val, np.array(val['scaledwordCount'])[:, None]))
    #
    # lr.fit(X, train['Sentiment'])
    #
    # model = Sequential()
    #
    # early_stopping_monitor = EarlyStopping(patience=1)
    #
    # model.add(Dense(64, input_shape=(1631,)))
    # model.add(Dense(128, activation="tanh"))
    # model.add(Dense(128, activation="tanh"))
    # model.add(Dense(64, activation="tanh"))
    #
    # model.add(Dense(3, activation='softmax'))
    # model.summary()
    #
    # # In[170]:
    #
    # opt = optimizers.Adam(0.0001)
    #
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # # In[171]:
    #
    # h_callback = model.fit(X, pd.get_dummies(train['Sentiment']), validation_split=0.1, epochs=12, batch_size=48,
    #                        callbacks=[early_stopping_monitor])
    #
    # # In[173]:
    #
    # predictions = []
    # predictions1 = model.predict(X_val)
    #
    # for i in range(len(X_val)):  # .ravel()
    #     predictions.append(np.argmax(predictions1[i]))
    #
    # def lrpredict(rev):
    #     testrew = vectorizer.transform(rev)
    #     testrew = scipy.sparse.hstack((testrew, np.array([len(rev)])[:, None])).A
    #     print("LogisticRegression predicts: ", labels[lr.predict(testrew)[0]])
    #     print(lr.predict_proba(testrew))
    #     return None
    #
    # def nnpredict(rev):
    #     testrew = vectorizer.transform(rev)
    #     testrew = scipy.sparse.hstack((testrew, np.array([len(rev)])[:, None])).A
    #     print("Neural Network predicts: ", labels[np.argmax([model.predict(testrew)])])
    #     print(model.predict(testrew))
    #     return None

    # lrpredict(description), nnpredict(description),












  # useridList = []
    # users = CustomerProfile.query.all()
    #
    # for user in users:
    #     useridList.append(user.Review)
    # # description = CustomerProfile.query.filter_by(Review=entry_id)
    # # test = []
    # # test.append(description)
    # return useridList

    # return render_template("add.html",description = description)
    # entry = CustomerProfile.query.get_or_404(int(entry_id))
    # rows = db.session.query(CustomerProfile). \
    #     with_entities(CustomerProfile.Review). \
    #             all()
    # as_dict = [dict(zip(row.keys(), row) for row in rows)]
    # [{('Review', 'this is ok'): ('Output', 'Negative')}]

# Creating data for correlation
# def heatmap(datax):
#     naming = str(datax.iat[1, 6])
#     correlationdata = datax[
#         ["Review Rating", "Type of Trip:", "No. of contributions from reviewer:", "No. of helpful votes on review:",
#          "Month"]]
#     correlationdata
#     if naming =="Yotel":
#         check = str(datax.iat[-1,6])
#         if check !="Yotel":
#             # Plotting correlation graph
#             fig = plt.figure(figsize=(5, 5))
#             heat_map = sns.heatmap(correlationdata.corr(), annot=True, cmap='PiYG')
#             plt.title("HeatMap to show correlation between attributes for Overall")
#             img = io.BytesIO()
#             fig.savefig(img)
#             img.seek(0)
#             return send_file(img, mimetype='img/png')
#         else:
#             # Plotting correlation graph
#             fig = plt.figure(figsize=(5, 5))
#             heat_map = sns.heatmap(correlationdata.corr(), annot=True, cmap='PiYG')
#             plt.title("HeatMap to show correlation between attributes for " + naming)
#             img = io.BytesIO()
#             fig.savefig(img)
#             img.seek(0)
#             return send_file(img, mimetype='img/png')
#     else:
#         # Plotting correlation graph
#         fig = plt.figure(figsize=(5, 5))
#         heat_map = sns.heatmap(correlationdata.corr(), annot=True, cmap='PiYG')
#         plt.title("HeatMap to show correlation between attributes for " + naming)
#         img = io.BytesIO()
#         fig.savefig(img)
#         img.seek(0)
#         return send_file(img, mimetype='img/png')

# Training the model
# Finding vif
# heatmap(Yotelcdata)
# heatmap(HotelGcdata)
# heatmap(ShangriLacdata)
# heatmap(Fullertoncdata)
# heatmap(Carltoncdata)
# heatmap(Overallcdata)

# find_vif(Yotelcdata)
# find_vif(HotelGcdata)
# find_vif(ShangriLacdata)
# find_vif(Fullertoncdata)
# find_vif(Carltoncdata)
# find_vif(Overallcdata)

#compute_ROC_AUC(Yotelcdata)

# percentage_pie(Yoteldata)
# percentage_pie(HotelGdata)
# percentage_pie(ShangriLadata)
# percentage_pie(Fullertondata)
# percentage_pie(Carltondata)
# percentage_pie(Overalldata)

#no_of_cust_over_year(Overallcdata)

#bar_review_rating(Overallcdata)

# from nltk.corpus import stopwords
# from wordcloud import WordCloud
# import nltk
#
# with app.app_context(df):
#     pos_review = [j for i, j in enumerate(df['Review Description']) if df['Sentiment'][i] == 2]
#     neu_review = [j for i, j in enumerate(df['Review Description']) if df['Sentiment'][i] == 1]
#     neg_review = [j for i, j in enumerate(df['Review Description']) if df['Sentiment'][i] == 0]
#
#     pos_review_title = [j for i, j in enumerate(df['Review Title']) if df['Sentiment'][i] == 2]
#     neu_review_title = [j for i, j in enumerate(df['Review Title']) if df['Sentiment'][i] == 1]
#     neg_review_title = [j for i, j in enumerate(df['Review Title']) if df['Sentiment'][i] == 0]
# @app.route('/trial')
# def trial():
#     # Create a Wordcloud from the reviews
#     nltk.download('stopwords')
#     stop_words = set(stopwords.words('english'))
#     stop_words.update(["hotel", "shangri la", "shangri", "la", "service", "singapore", "room", "rooms", "like", "staff",
#                        "time"])  # all these words are removed because they were included in negative sentiment too which contradicts
#     ShangriLaSA = "./static/data/ShangriLa_SA_updated"
#     df = pd.read_csv(ShangriLaSA)
#     pos_review = [j for i, j in enumerate(df['Review Description']) if df['Sentiment'][i] == 2]
#     all_words = ' '.join([text for text in pos_review])
#     wordcloud = WordCloud(stopwords=stop_words, width=800, height=500, background_color="white", random_state=21,
#                           max_font_size=110).generate(all_words)
#
#     plt.figure(figsize=(10, 7))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis('off')
#     plt.show()
#     return render_template('trial.html')





# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import io
# import base64
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import seaborn as sns
# import matplotlib.pylab as plt
# #@app.route('/plot.png')
# #def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure():
#     # Retrieving dataset
#     data = pd.read_csv('test.csv')
#
#     # Look at data, type and check for NAs
#     data.info()
#
#     # Remove NA values
#     data = data.dropna()
#
#     # Replace 'Type of Trip:' to numbers to represent each type of the trip
#
#     # Creating data for correlation
#     correlationdata = data[
#         ["Review Rating", "Type of Trip:", "No. of contributions from reviewer:", "No. of helpful votes on review:"]]
#     correlationdata
#     correlationdata.corr()
#
#     # Plotting correlation graph
#     plt.figure(figsize=(4, 4))
#     heat_map = sns.heatmap(correlationdata.corr(), annot=True, cmap='PiYG')
#     plt.title("HeatMap using Seaborn Method")
#     return plt.show()
# @app.route('/visualize')
# def visualize():
#     sns.heatmap(correlationdata.corr(), annot=True, cmap='PiYG')
#     canvas = FigureCanvas(fig)
#     fig = create_figure()
#     output = io.BytesIO()
#     img = io.BytesIO
#     canvas.print_png(output)
#     fig.savefig(img)
#     img.seek(0)
#     return send_file(img,mimetype='img/png')

# plt.style.use("seaborn")
# fig,ax = plt.subplots(figsize=(4, 4))
# ax= sns.set_style(style="darkgrid")
# excel_data1 = "./static/data/heatmap.csv"
# data = pd.read_csv(excel_data1)
# data = data.dropna()
#
# data = data.replace(to_replace="Travelled solo",
#                         value="1")
# data = data.replace(to_replace="Travelled as a couple",
#                         value="2")
# data = data.replace(to_replace="Travelled with friends",
#                         value="3")
# data = data.replace(to_replace="Travelled with family",
#                         value="4")
# data = data.replace(to_replace="Travelled on business",
#                         value="5")
#
#     # Changing data type to int
# data['Type of Trip:'] = data['Type of Trip:'].astype('int')
#
# correlationdata = data[
#         ["Review Rating", "Type of Trip:", "No. of contributions from reviewer:", "No. of helpful votes on review:"]]

#Editted graphs out:
 # Creating the graphs of Review rating against No. of helpful votes based on the type of trips - Fullerton
    # (To show which category of tourist contributes the most to review rating and votes)
    # fig1 = px.bar(df, x="Review Rating", y="No. of helpful votes on review:", color="Type of Trip:",
    #              title='Graph of review rating and no. of helpful votes')
    # Bar1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    #
    # # Creating the histogram based on No. of contributions from reviewer and type of trips
    # fig2 = px.histogram(df, x="No. of contributions from reviewer:", y="Type of Trip:", color="Review Rating")
    # Histogram1 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

 # #Creating the first graph between Review rating and No. of helpful votes on review using histograms - Comparison
    # fig = px.histogram(Overallcdata, x="Review Rating", y="No. of helpful votes on review:", color="Hotels")
    # hist = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #
    # #Creating the second graph between Review rating and No. of contributions from reviewer
    # fig1 = px.histogram(Overallcdata, x="Review Rating", y="No. of contributions from reviewer:", color="Hotels")
    # hist1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

# return render_template("CarltonMain.html", graphJSON=test, description=description)





#Creating a table graph to showcase abbreviation and its description
# values = [['<b>VIF<br>(Variance Inflation<br>Factor)</b>', '<b>RMSE<br>(Root Mean <br>Square Error)</b>', '<b>ROC Curve<br>(Receiver Operating <br>Characteristic Curve)</b>', '<b>AUC<br>(Area Under the<br>ROC Curve)</b>'], #1st col
  # ["A measure to detect any multicollinearity and its severity between independent variables. It is essential to compute VIF to ensure there are no inputs that are influencing each other, which might potentially cause the prediction model to be inaccurate. Multicollinearity is present when any of the value is more than 10.",
  # "A measure to compute the standard deviation and the spread of the data points from the regression line. A lower RMSE value (usually less than 1) is considered a good indicator.",
  # "A graph to illustrate the performance of the predictive model in classifying at different thresholds. The graph is plotted with two parameters, 'True Positive Rate' and 'False Positive Rate'. The higher the arc of the curve, the better the model is at classifying true positive.",
  # "A measure to compute the area under the curve created by the ROC Curve. AUC ranges from 0 to 1, where 1 indicates that the model is able to predict 100% correctly and 0 indicates 100% wrong prediction. An acceptable value of AUC is more than 0.5."]]

#Computing correlation

 # def prediction(model, text_array, item_list):
    #     prob = model.predict(X)
    #     pro_df = pd.DataFrame(prob, columns=item_list)
    #     result = item_list[np.argmax(prob)]
    #     return pro_df, result
    #
    # sequences = tokenizer.texts_to_sequences(text_input)
    #
    # X = pad_sequences(sequences, maxlen=max_len)
    #
    # y = np.asarray(sent)
    #
    # # Load the ratings_dictionary
    # df_map = pd.read_csv('./static/data/mapping.csv', delimiter=',')
    # sent_dictionary = df_map.loc[:, 'sentiment'].to_dict()
    #
    # def prediction(model, text_input, items_l):
    #     sequences = tokenizer.texts_to_sequences(text_input)
    #     data = pad_sequences(sequences, maxlen=max_len)
    #     prob = model.predict(data)
    #     pro_df = pd.DataFrame(prob, columns=items_l)
    #     result = items_l[np.argmax(prob)]
    #     return pro_df, result


# import random as rand
# import os
# import seaborn as sns
# import matplotlib as mp
# import matplotlib.pyplot as plt
# import matplotlib.pylab as plt

# from keras import Sequential, optimizers
# from keras.callbacks import EarlyStopping
# from keras.layers import Embedding, GRU, Dropout, Dense
# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
# from keras_preprocessing.sequence import pad_sequences
# from nltk import word_tokenize
# from nltk.corpus import stopwords


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import io
# import statsmodels.api as sm
# import datetime
# from tensorflow.python.data.util import sparse