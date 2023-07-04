import streamlit as st
import json
import numpy as np
from scipy.stats import norm,bernoulli,gamma
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly.express as px
import plotly.graph_objects as go



PATH_CONFIG = "configs.json"

# Load configs
try:
    with open(PATH_CONFIG,"r") as f:
        dict_conf = json.load(f)
except:
    st.write("Error occured in reading config files.")
    st.stop()


def f_save_config():
    """ Save the config files, to have the previouse session data inorder not to loss them"""
    try:
        with open(PATH_CONFIG,"w+") as f:
            json.dump(dict_conf, f)
    except:
        st.write("Error occured in reading config files.")
        st.stop()

def f_onchange_slider1():
    """ When each parameter changes, the oppsite parameters should save its own data to retrieve last state"""
    dict_conf["x2_mean"] = st.session_state.x2_mean_slider
    dict_conf["x2_std"] =  st.session_state.x2_std_slider
    dict_conf["x1_seed"] = np.random.randint(1,10000)
    f_save_config()
def f_onchange_slider2():
    """ When each parameter changes, the oppsite parameters should save its own data to retrieve last state"""
    dict_conf["x1_mean"] = st.session_state.x1_mean_slider
    dict_conf["x1_std"] =  st.session_state.x1_std_slider
    dict_conf["x2_seed"] = np.random.randint(1,10000)
    f_save_config()


st.markdown("## Feature Distribution")
st.markdown("This is a simple playground to play with different feature distribution to see\
            what will happen to score of the model when distribution changes.")


n_samples= 300

col1, col2 = st.columns(2,gap="medium")
with col1:
    st.header("x1")
    x1_mean = st.slider('Mean', -5., 5., value=0., step=.2, key="x1_mean_slider", on_change=f_onchange_slider1)
    x1_std = st.slider('STD', 0., 3., value=1., step=.2, key="x1_std_slider", on_change=f_onchange_slider1)

with col2:
    st.header("x2")
    x2_mean = st.slider('Mean', -5., 5., value=0., step=.2, key="x2_mean_slider",on_change=f_onchange_slider2)
    x2_std = st.slider('STD', 0., 3., value=1., step=.2, key="x2_std_slider",on_change=f_onchange_slider2)


# make data
np.random.seed(dict_conf["x1_seed"])
x1 = norm.rvs(loc=x1_mean, scale=x1_std, size=n_samples)
np.random.seed(dict_conf["x2_seed"])
x2 = norm.rvs(loc=x2_mean, scale=x2_std, size=n_samples)

x = np.concatenate((x1,x2))
y = np.hstack(( np.full_like(x1, 0, dtype=np.int32), np.full_like(x2, 1, dtype=np.int32)))

# build dataset
data = np.stack((x,y), axis=1)
np.random.shuffle(data)

# create train-test
n_test = int(data.shape[0]*.9)
data_train = data[:-n_test]
data_test = data[-n_test:]


fig = go.Figure()
fig.update_xaxes(range=[-10, 10])
fig.update_yaxes(range=[0, 100])
fig.add_trace(go.Histogram(x=x1, xbins=dict( start=-5, end=5, size=.5 ),
    marker_color='blue',
    opacity=0.5,
    name="x1"))
fig.add_trace(go.Histogram(x=x2,xbins=dict( start=-5, end=5, size=.5 ),
    marker_color='red',
    opacity=0.5,
    name="x2"))

fig.update_layout(barmode='overlay')
st.plotly_chart(fig)


if st.button('Compare Results') == False:
    st.stop()

## SVC model
dict_results = {}
model_svc = SVC()
model_svc.fit(data_train[:,0].reshape(-1,1), data_train[:,1])
score = model_svc.score(data_test[:,0].reshape(-1,1), data_test[:,1])
dict_results["SVC"] = score*100

## Logistic Reg model
model_lreg = LogisticRegression()
model_lreg.fit(data_train[:,0].reshape(-1,1), data_train[:,1])
model_lreg.score(data_test[:,0].reshape(-1,1), data_test[:,1])
score = model_lreg.score(data_test[:,0].reshape(-1,1), data_test[:,1])
dict_results["Log. Reg"] = score*100

## NN Model
model_mlp = MLPClassifier(hidden_layer_sizes=10, activation="relu", max_iter=20, alpha=.001,)
model_mlp.fit(data_train[:,0].reshape(-1,1), data_train[:,1])
score = model_mlp.score(data_test[:,0].reshape(-1,1), data_test[:,1])
dict_results["NN"] = score*100


fig = go.Figure(data=go.Bar(x=list(dict_results.keys()), y=list(dict_results.values()),marker_color="green"))
fig.update_layout(
       title='Scores',
       xaxis_title='Models',
       yaxis_title='Score')
fig.update_yaxes(range=[0, 100])
st.plotly_chart(fig)
