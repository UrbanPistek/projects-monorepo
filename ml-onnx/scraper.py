import io
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from rapidfuzz import fuzz


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


def plot_models_table(df: pd.DataFrame):

    # Set seaborn theme
    sns.set_theme(style="whitegrid", context="talk")

    # Create the scatterplot
    plt.figure(figsize=(16, 9))
    scatter = sns.scatterplot(data=df, x='Params', y='Acc@1', s=60, color='dodgerblue', edgecolor='black', alpha=0.7)

    # Annotate a subset of points to avoid clutter
    top_annotate = df.nlargest(5, 'Acc@1')
    bottom_annotate = df.nsmallest(5, 'Acc@1')
    r_annotate = df.nsmallest(5, 'acc_per_params_M')

    for _, row in top_annotate.iterrows():
        plt.text(row['Params'], row['Acc@1'], row['Weight'], fontsize=9, weight='bold', alpha=0.8)
    for _, row in bottom_annotate.iterrows():
        plt.text(row['Params'], row['Acc@1'], row['Weight'], fontsize=9, weight='bold', alpha=0.8)
    for _, row in r_annotate.iterrows():
        plt.text(row['Params'], row['Acc@1'], row['Weight'], fontsize=9, weight='bold', alpha=0.8)

    # Improve layout
    plt.title('Model Accuracy vs Parameters', fontsize=16)
    plt.xlabel('Params (millions)', fontsize=14)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=14)
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

url = "https://docs.pytorch.org/vision/main/models.html#table-of-all-available-classification-weights"
df = get_data_table_from_url(url)
print(df.head())

#
# df['Weight'] = df['Weight'].str.replace('_Weights.IMAGENET1K_V1', '', regex=False)
df['Weight'] = df['Weight'].apply(lambda x: replace_similar_substrings(x, "IMAGENET1K_V1", 50))
df['Weight'] = df['Weight'].str.replace('_Weights', '', regex=False)

# Get acc / params ratio
df['Params'] = df['Params'].str.replace('M', '', regex=False).astype(float) # remove M and convert to float
df["acc_per_params_M"] = df["Acc@1"] / df["Params"]
top5 = df.nlargest(5, 'acc_per_params_M')
print(top5)

plot_models_table(df)
