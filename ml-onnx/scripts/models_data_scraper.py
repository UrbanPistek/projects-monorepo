import io
import os
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


def plot_models_table(
        df: pd.DataFrame, 
        xcol: str, 
        ycol: str, 
        label: str,
        title: str, 
        xaxis: str,
        yaxis: str
    ):

    # Set seaborn theme
    sns.set_theme(style="whitegrid", context="talk")

    # Create the scatterplot
    plt.figure(figsize=(16, 9))

    # Annotate a subset of points to avoid clutter
    annotate_acc = df[df['Acc@1'] > 85]

    # Get unique names
    unique_names = df['model_type'].unique()
    print("Unique model types:", unique_names)

    # Get rows with highest and lowest scores for each name
    def get_min_max_rows(group):
        max_idx = group['Acc@1'].idxmax()
        min_idx = group['Acc@1'].idxmin()
        
        # If min and max are the same (only one row), return just one row
        if max_idx == min_idx:
            return group.loc[[max_idx]]
        else:
            return group.loc[[max_idx, min_idx]]

    result = (
        df.groupby('model_type', group_keys=False)
        .apply(get_min_max_rows, include_groups=False)
        .reset_index(drop=True)
    )

    # Add scatter hue based on model types
    palette = sns.color_palette("tab20", n_colors=len(unique_names))
    scatter = sns.scatterplot(data=df, x=xcol, y=ycol, s=60, hue="model_type", palette=palette, edgecolor='black', alpha=0.7)
    
    # Keep only rows unique to result
    merged = result.merge(annotate_acc, how="left", indicator=True)
    annotate = merged[merged["_merge"] == "left_only"].drop(columns="_merge")

    # Create 2 sets of texts to emphasize certain features more in 1 set
    texts = []
    for i, row in annotate.iterrows():
        texts.append(plt.text(row[xcol], row[ycol], row[label], fontsize=8, weight='normal', alpha=1))
    for i, row in annotate_acc.iterrows():
        texts.append(plt.text(row[xcol], row[ycol], row[label], fontsize=8, weight='bold', alpha=1))

    # Automatically adjust text to reduce overlap
    adjust_text(
        texts, 
        expand_points=(2.0, 2.0),  # expand distance from points (x, y)
        expand_text=(1.5, 1.5),    # expand distance from other texts
        force_text=(0.5, 0.5),     # strength of repulsion between texts
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.75), 
        iter_lim=1000
    )

    # Improve layout
    plt.title(title, fontsize=16)
    plt.xlabel(xaxis, fontsize=14)
    plt.ylabel(yaxis, fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(
        title="Model Groups", 
        ncol=3, # number of columns in the legend
        frameon=True,
        fontsize=8, 
        title_fontsize=10
    )
    plt.savefig(f"{title.replace(' ', '_')}", dpi=300, bbox_inches='tight')
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

    # Extract the name part before the underscore
    df['model_type'] = df['Weight'].str.split('_').str[0].str.replace(r'\d+', '', regex=True).str.strip()

    plot_models_table(df, xcol='Params', ycol='Acc@1', label='Weight', title='Params (M) v Acc@1', xaxis='Params (M)', yaxis='Acc@1')
    plot_models_table(df, xcol='GFLOPS', ycol='Acc@1', label='Weight', title='GFLOPS v Acc@1', xaxis='GFLOPS', yaxis='Acc@1')

    # Save
    if not os.path.exists("../data"):
        os.makedirs("../data")
    df.to_csv('../data/pytorch_models_data.csv')

main(URL)
