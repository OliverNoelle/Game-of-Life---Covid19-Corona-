""" 
-----------------------------------------------
Game of life - Corona Covid 19 Edition 
    
Author : Oliver Nölle, 2020 Hamburg/ Germany
Architecture  :
Python, Plotly/Dash   

Description :
Simulation of course of infection with game of life approach.
Members of a given population size are moving randomly on a playground (100X100 matrix).
If they meet on the same field they can infect each other. Infected will recover afer a given time.

Prediction model : 
SIR model. Parameters for differential equations are calculated based on best fit curve for 
infected data points  

Population groups : 
Susceptible (green) : group members can be infected
Infected (red): currently infected and can infect others located on the same field on the playground
Recovered (blue): members were infected and noew recovered or dead, Not infectous anymore
----------------------------------------------- 
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go # or plotly.express as px
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import pandas as pd
import time
from scipy.integrate import odeint
from scipy import integrate, optimize
from Habitant import habitant

pop_size = 100
country_x = 100
country_y = 100
initial_infected = 3
time_to_recover = 100
showsir=False
person = [habitant(country_x, country_y) for dummy in range(pop_size)]
    
for b in range(initial_infected):
        person[b].condition = 1

infected = initial_infected
susceptibel = pop_size-infected
recovered = 0   

healthmatrix = np.zeros([country_x, country_y], dtype = int)
df = pd.DataFrame([[0,susceptibel,infected,recovered]],columns = ['step', 'susceptibel','infected','recovered'])
dfloc = pd.DataFrame([[0,0,0]],columns = ['xpos', 'ypos','condition'])


def simstep(step, person): 
    global df
    global dfsim
    global healthmatrix
    global dfloc
    global pop_size
    population = pop_size
    country_x = 100
    country_y = 100
    time_to_recover = 100
    susceptibel = df['susceptibel'].iloc[len(df)-1] # letzte Wert des df
    infected = df['infected'].iloc[len(df)-1]
    recovered = df['recovered'].iloc[len(df)-1]
    location = np.array([0,0,0])

    for j in range (population):
            xpos = person[j].x_value
            ypos = person[j].y_value
            person[j].move(country_x, country_y)
            row = [xpos,ypos,person[j].condition]
            location = np.vstack((location, row))
        
            if person[j].condition == 1:
                person[j].infectticker += 1
                
            if person[j].condition == 1 and person[j].infectticker >= time_to_recover:
                person[j].condition = 3
                healthmatrix[xpos, ypos] = 0
                recovered += 1
                infected -= 1
        
            if person[j].condition == 1 :
                healthmatrix[person[j].x_value, person[j].y_value] = 1
                healthmatrix[xpos, ypos] = 0
            
            if person[j].condition == 0 and (healthmatrix[person[j].x_value, person[j].y_value] == 1 or healthmatrix[xpos, ypos] == 1):
                person[j].condition = 1  
                infected += 1
                susceptibel -= 1

    newrow = [step, susceptibel,infected,recovered]
    df.loc[len(df)] = newrow

    dfloc=pd.DataFrame(location, columns=['xpos','ypos','condition'])

    print(step)

 #   time.sleep(0.01)

# ------------ SIR Model --------------------------------------------------
def SIR_model(y,t,beta,gamma):
    S, I, R = y
    global pop_size
    global time_to_recover
    N=pop_size
    dS_dt = -1*beta*I*S/N 
    dI_dt = (beta*I*S/N) - gamma*I
    
    if t > time_to_recover : 
        dR_dt = gamma*I
    else : 
        dR_dt = 0
    
    return ([dS_dt, dI_dt, dR_dt])

def fit_odeint(x, beta, gamma):
    global pop_size
    global initial_infected
    suspected = pop_size - initial_infected
    return integrate.odeint(SIR_model, (suspected, initial_infected, 0), x, args=(beta, gamma))[:,1]

# --------------------------------------------------------------------------

def update_fig(df) :
    global showsir
    global sol
    global pop_size
    fig = go.Figure(data=go.Scatter(x=df['step'],
                                y=df['recovered'],
                                mode='lines',
                                name='recovered'
                                )) # hover text goes here

    fig.add_trace(go.Scatter(x=df['step'], y=df['infected'],
                    mode='lines',
                    name='infected'))
    fig.add_trace(go.Scatter(x=df['step'], y=df['susceptibel'],
                    mode='lines',
                    name='susceptibel'))

    if showsir == True :
            t = np.linspace(0,1000,1000)
            new_susp=pop_size-sol[:,1]-sol[:,2]
            fig.add_trace(go.Scatter(x=t, y=sol[:,1],
                    mode='lines',
                    name='SIR infected'))
            fig.add_trace(go.Scatter(x=t, y=sol[:,2],
                    mode='lines',
                    name='SIR recovered'))   
            fig.add_trace(go.Scatter(x=t, y=new_susp,
                    mode='lines',
                    name='SIR suspected'))        

    fig.update_layout(title='Course of Infection',xaxis= dict(title= 'simulation step',
                    range=[0, 1000], autorange=False))
    fig.update_layout(autosize=False,width=800,height=500)

    return fig

def update_fig2(dfloc) :
    dfloc_s = dfloc.loc[dfloc['condition']==0]
    dfloc_i = dfloc.loc[dfloc['condition']==1]
    dfloc_r = dfloc.loc[dfloc['condition']==3]
 
    fig2 = go.Figure(data=go.Scatter(x=dfloc_s['xpos'],
                                y=dfloc_s['ypos'],
                                mode='markers',
                                marker=dict(size=16,color='green'),
                                name='susceptibel'
                                )) 
    fig2.add_trace(go.Scatter(x=dfloc_i['xpos'],
                                y=dfloc_i['ypos'],
                                mode='markers',
                                marker=dict(size=16,color='red'),
                                name='infected'
                                )) 
    fig2 .add_trace(go.Scatter(x=dfloc_r['xpos'],
                                y=dfloc_r['ypos'],
                                mode='markers',
                                marker=dict(size=16,color='blue'),
                                name='recovered'
                                )) 
    
    
    fig2.update_layout(xaxis= dict(title= '',range=[0, 100], autorange=False, showline=False,
                                showgrid=False,showticklabels=False))
    fig2.update_layout(yaxis= dict(title= '',range=[0, 100], autorange=False, showline=False,
                                showgrid=False, showticklabels=False))
    fig2.update_layout(title='Playground - Population',autosize=False,width=600,height=500)

    return fig2

fig=update_fig(df)
fig2=update_fig2(dfloc)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div(children=[
        html.Button('Reset parameters', id='setvalues', n_clicks=0),
        html.Button('Start simulation', id='restart', n_clicks=0),
        html.Button('Stop simulation', id='stopsim', n_clicks=0),
        html.Button('SIR model', id='sirmodel', n_clicks=0),
#        html.A(html.Button('Refresh screen'),href='/'), #Button for refrsh data
        html.Div(dcc.Input(id='population', value=pop_size, type='text')),
        html.Div(id='button-population',children='population'),
        html.Div(dcc.Input(id='infected', value=initial_infected, type='text')),
        html.Div(id='button-infected',children='infected'),
        html.Div(
                dcc.Graph(id='linechart',figure=fig), 
                style={'display': 'inline-block'}
                ), 
                dcc.Interval(
                    id='interval-component',
                    interval=1000, # in milliseconds
                    max_intervals=1000,
                    disabled = True,
                    n_intervals=0
                ),  
        html.Div(
                dcc.Graph(id='playground',figure=fig2), 
                style={'display': 'inline-block'}
                )

        ], style={'width': '100%', 'display': 'inline-block'}) 


# https://community.plotly.com/t/how-to-turn-off-an-interval/16838 : turn off interval
@app.callback(
            Output('interval-component', 'disabled'),
            [Input('restart', 'n_clicks'),
            Input('stopsim', 'n_clicks'),
            Input('setvalues', 'n_clicks'),
            Input('sirmodel','n_clicks')],
            [State('population', 'value')])
def start_simulation(n,m,o,q,p):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if n==0 :   # prevent trigger when button is not clicked - n_clicks = 0
        return True
    elif 'restart' in changed_id:
        return False
    elif 'stopsim' in changed_id:
        return True
    elif 'setvalues' in changed_id:
        return True
    elif 'sirmodel' in changed_id:
        return True

@app.callback(
            Output('interval-component', 'n_intervals'),
            [Input('setvalues', 'n_clicks')],
            [State('population', 'value'),
             State('infected','value')])
def reset_simulation(clicks, popsize, infsize):
    global df
    global dfsim
    global healthmatrix
    global dfloc
    global pop_size 
    global person
    global showsir

    if clicks>0:
        pop_size = int(popsize)
        initial_infected = int(infsize)
        person = [habitant(country_x, country_y) for dummy in range(pop_size)]
        for b in range(initial_infected):
                person[b].condition = 1

        infected = initial_infected
        susceptibel = pop_size-infected
        recovered = 0   
        showsir = False

        healthmatrix = np.zeros([country_x, country_y], dtype = int)
        df = pd.DataFrame([[0,susceptibel,infected,recovered]],columns = ['step', 'susceptibel','infected','recovered'])
        dfloc = pd.DataFrame([[0,0,0]],columns = ['xpos', 'ypos','condition'])
    return 0
 


@app.callback(Output('linechart', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('sirmodel', 'n_clicks')])
def update_graph_live(n,m):
            global sol
            global pop_size
            global initial_infected
            global showsir
            simstep(n, person)
            fig=update_fig(df)

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'sirmodel' in changed_id and m>0 and n>10:  #calculate SIR model not before 10 simulation steps
                print ('SIR simulation')
                xdata = df.step
                ydata = df.infected
                xdata = np.array(xdata, dtype=float)
                ydata = np.array(ydata, dtype=float)

                S0 = pop_size-initial_infected
                I0 = initial_infected
                R0 = 0
                y = S0, I0, R0

                popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
                fitted = fit_odeint(xdata, *popt)
                bta = popt[0]
                gmma = popt[1]
                t = np.linspace(0,1000,1000)
                sol = odeint(SIR_model,[S0,I0,R0],t,args = (bta,gmma))
                sol = np.array(sol)
                showsir=True
 #               print(sol[:,1])
                fig=update_fig(df)
            return fig      

@app.callback(Output('playground', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
            fig2=update_fig2(dfloc)
            return fig2  


if __name__ == '__main__': 
    app.run_server(debug=True)
