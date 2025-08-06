import io
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from adjustText import adjust_text

URL = "https://docs.pytorch.org/vision/main/models.html#table-of-all-available-classification-weights"

def get_data_table_from_url(url: str) -> pd.DataFrame:

    # Initialize return
    df = None

    # Get data from url
    response = requests.get(url)
    if response.status_code == 200:

        # Parse html from the response
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all <table> elements
        tables = soup.find_all('table')
        table = tables[1] # After some looking, its the 2nd table that I want
        table_string = io.StringIO(str(table)) # pandas works on html string input

        # We only need the first result for this problem
        tables: list = pd.read_html(table_string)
        df: pd.DataFrame = pd.DataFrame(tables[0])

    return df


def plot_models_table(df: pd.DataFrame, xcol: str, ycol: str, label: str):

    # Set seaborn theme
    sns.set_theme(style="whitegrid", context="talk")

    # Create the scatterplot
    plt.figure(figsize=(16, 9))
    scatter = sns.scatterplot(data=df, x=xcol, y=ycol, s=60, color='dodgerblue', edgecolor='black', alpha=0.7)

    # Annotate a subset of points to avoid clutter
    annotate = df[df['Acc@1'] > 85]
    texts = []
    for i, row in annotate.iterrows():
        texts.append(plt.text(row[xcol], row[ycol], row[label], fontsize=9, weight='bold', alpha=0.8))

    # Automatically adjust text to reduce overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Improve layout
    plt.title(f'{ycol} vs {xcol}', fontsize=16)
    plt.xlabel(xcol, fontsize=14)
    plt.ylabel(ycol, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def replace_similar_substrings(text: str, target: str, threshold=80):
    words = text.split(".")
    new_words = [
        "" if fuzz.ratio(word.lower(), target.lower()) >= threshold else word
        for word in words
    ]
    return ' '.join(new_words)


def main(url: str):

    df = get_data_table_from_url(url)
    print(df.head())

    # Clean up names 
    df['Weight'] = df['Weight'].apply(lambda x: replace_similar_substrings(x, "IMAGENET1K_V1", 50))
    df['Weight'] = df['Weight'].str.replace('_Weights', '', regex=False)

    # Get acc / params ratio
    df['Params'] = df['Params'].str.replace('M', '', regex=False).astype(float) # remove M and convert to float
    df["acc_per_params_M"] = df["Acc@1"] / df["Params"]
    top5 = df.nlargest(5, 'acc_per_params_M')
    print(top5)

    plot_models_table(df, xcol='Params', ycol='Acc@1', label='Weight')
    plot_models_table(df, xcol='GFLOPS', ycol='Acc@1', label='Weight')

    # Save
    df.to_csv('data/pytorch_models_data.csv')

main(URL)
