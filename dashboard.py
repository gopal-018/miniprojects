#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


uploaded_data = pd.read_csv("Heart.csv")


column_options = [{"label": col, "value": col} for col in uploaded_data.columns]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div([
    dbc.Container([
        html.H1("Heart Disease Dataset Analysis", className="text-center mt-4"),
        
        dbc.Row([
         
            dbc.Col([
                html.H5("Select Visualization Type", className="mt-3"),
                dcc.Dropdown(
                    id="visualization-selector",
                    options=[
                        {"label": "Scatter Plot", "value": "scatter"},
                        {"label": "Pie Chart", "value": "pie"},
                        {"label": "Box Plot", "value": "box"},
                        {"label": "Heatmap", "value": "heatmap"},
                        {"label": "Density Plot", "value": "density"},
                        {"label": "Line Plot", "value": "line"},
                        {"label": "Bar Graph", "value": "bar"}
                    ],
                    placeholder="Select a visualization type",
                    value=None,
                    style={"margin-bottom": "20px"}
                )
            ], width=3),

       
            dbc.Col([
                html.H5("Select Attributes", className="mt-3"),
                dcc.Dropdown(
                    id="x-axis-selector",
                    options=column_options,
                    placeholder="Select X-Axis Attribute",
                    value=None,
                    style={"margin-bottom": "10px"}
                ),
                dcc.Dropdown(
                    id="y-axis-selector",
                    options=column_options,
                    placeholder="Select Y-Axis Attribute (if applicable)",
                    value=None,
                    style={"margin-bottom": "10px"}
                ),
                dcc.Dropdown(
                    id="group-selector",
                    options=column_options,
                    placeholder="Select Group/Color Attribute (if applicable)",
                    value=None,
                    style={"margin-bottom": "10px"}
                )
            ], width=3),


            dbc.Col([
                dcc.Graph(id='visualization-output', style={'height': '500px'})
            ], width=6)
        ])
    ])
])

@app.callback(
    Output('visualization-output', 'figure'),
    [
        Input('visualization-selector', 'value'),
        Input('x-axis-selector', 'value'),
        Input('y-axis-selector', 'value'),
        Input('group-selector', 'value')
    ]
)
def update_visualization(visualization_type, x_axis, y_axis, group):
    df = uploaded_data

 
    default_fig = go.Figure()
    default_fig.update_layout(
        title="Select a visualization type and attributes to display.",
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0"
    )
    
    if visualization_type is None:
        return default_fig
    
  
    layout_settings = {
        "paper_bgcolor": "#f0f0f0",
        "plot_bgcolor": "#f0f0f0"
    }

    if visualization_type == 'scatter':
        if x_axis and y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=group, title=f"Scatter Plot: {x_axis} vs {y_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'pie':
        if x_axis:
            fig = px.pie(df, names=x_axis, values=group, title=f"Pie Chart of {x_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'box':
        if x_axis and y_axis:
            fig = px.box(df, x=x_axis, y=y_axis, color=group, title=f"Box Plot of {y_axis} by {x_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'heatmap':
        if set(df.columns).issuperset([x_axis, y_axis]):
            corr = df.corr()
            fig = px.imshow(corr, title="Heatmap of Correlations", color_continuous_scale='viridis')
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'density':
        if x_axis and y_axis:
            fig = px.density_contour(df, x=x_axis, y=y_axis, color=group, title=f"Density Plot: {x_axis} vs {y_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'line':
        if x_axis and y_axis:
            fig = px.line(df, x=x_axis, y=y_axis, color=group, title=f"Line Plot: {x_axis} vs {y_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    elif visualization_type == 'bar':
        if x_axis and y_axis:
            fig = px.bar(df, x=x_axis, y=y_axis, color=group, title=f"Bar Graph: {x_axis} vs {y_axis}")
            fig.update_layout(**layout_settings)
            return fig
        else:
            return default_fig

    return default_fig


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




