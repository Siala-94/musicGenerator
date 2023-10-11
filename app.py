from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

########## Functions ##########


def generate(input_text):
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")
    generated_output = model.generate(
        encoded_input, max_length=700, num_beams=7, early_stopping=True)
    generated_lyrics = tokenizer.decode(
        generated_output[0], skip_special_tokens=True)
    return generated_lyrics

########## Model definition ##########


PEFT_MODEL = "siala94/bert-lyrics-generator"
model = AutoModelForSeq2SeqLM.from_pretrained(PEFT_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL)

########## Dash app ##########

app = Dash(__name__)

app.layout = html.Div([
    # Container to center the elements and apply flexbox
    html.Div([
        html.H6("Input the song details"),

        # Input fields
        dcc.Input(id='input-title', type='text', placeholder='Song Title',
                  style={'width': '20%', 'marginRight': '10px'}),
        dcc.Input(id='input-artist', type='text', placeholder='Artist',
                  style={'width': '20%', 'marginRight': '10px'}),
        dcc.Input(id='input-genre', type='text',
                  placeholder='Genre', style={'width': '20%'}),

        # Button centered
        html.Div([
            html.Button(id='submit-button', n_clicks=0,
                        children='Generate Lyrics')
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px'}),

    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'gap': '10px'}),

    # Output div with 30% width, centered
    html.Div(id='output', children='', style={
             'width': '30%', 'margin': '20px auto'})
])


@app.callback(
    Output('output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-title', 'value'),
     State('input-artist', 'value'),
     State('input-genre', 'value')]
)
def update_output(n_clicks, songtitle, artist, genre):
    if n_clicks > 0:
        combined_input = f"{songtitle}{artist}{genre}"
        return generate(combined_input)
    return ''  # Default value


if __name__ == '__main__':
    app.run_server(debug=True)
