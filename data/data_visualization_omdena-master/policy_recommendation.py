import dash
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_core_components as dcc
#import dash_auth
import plotly.graph_objs as go
import pandas as pd
import dash_table
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from newsplease import NewsPlease
import pickle

gs = pd.read_csv('data/gold_standard.csv')
policy_df = pd.read_csv('data/policy_df.csv')
def load_obj(month, idx):
    '''Used to load the pickle article using
    month and idx parameters of gold standard row'''
    month = str(month).zfill(2)
    idx = str(idx).zfill(5)
    with open("data/texts/{}/{}.pkl".format(month, idx), "rb") as f:
        return pickle.load(f)

gs_articles = {}

for i in range(len(gs)):
    article = load_obj(gs['month'][i], gs['ids'][i])
    gs_articles[i] = article



app = dash.Dash()

app.layout = html.Div(children=[html.H3('Gold Standard Data'),
dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in gs.columns],
    data=gs.to_dict('records'),
    row_selectable="single",
    selected_rows = [],
    fixed_rows={ 'headers': True, 'data': 0 },
    style_table={'maxHeight': '300px','overflowX': 'scroll'},
    style_cell={
        'textAlign': 'left',
        'height': 'auto',
        'minWidth': '0px', 'maxWidth': '180px',
        'whiteSpace': 'normal'
    }
), html.Br(), html.Div(id='article-div'),
html.H3("Policy recommendations based on consine similarity are :"),
html.Div(id='recomend-div')
], className='container')

@app.callback(
    Output('article-div', "children"),
    [Input('table', "selected_rows")])
def update_article(selected_rows):
    #print(f"{gs_articles[0]}")
    if len(selected_rows):
        return f"selected row {selected_rows}, article is {gs_articles[selected_rows[0]].text}"
    else :
        return f"Please select one of the above row to get the policy recommendation"


@app.callback(
    Output('recomend-div','children'),
    [Input('table', "selected_rows")])
def recommend_policies(selected_rows):
  if len(selected_rows):
      article = gs_articles[selected_rows[0]]
      policies_recommended = []
      # policy_text_arrays = policy_df['policy_text'].values
      policy_text_arrays = policy_df['policy_text'].values
      article_index = len(policy_text_arrays)
      policy_text_arrays = np.append(policy_text_arrays, article.text)
      tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
      tfidf_matrix = tfidf.fit_transform(policy_text_arrays)
      cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
      cosine_values_series = pd.Series(cosine_similarities[article_index]).sort_values(ascending=False)
      top_10_indexes = list(cosine_values_series.iloc[1:5].index)

      for i in top_10_indexes:
        #policies_recommended.append((policy_df['policy_name'][i], cosine_values_series[i]))
        policies_recommended.append(policy_df['policy_name'][i])

      # recomend_df = pd.DataFrame(policies_recommended[:5], columns=['policy_names in order', 'cosine similarity'])

      #return recomend_df.to_dict('records')
      return f"{policies_recommended}"



if __name__ == '__main__':
    app.run_server(debug=True)
