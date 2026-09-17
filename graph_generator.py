import os
import pandas as pd
import matplotlib.pyplot as plt
from database import get_training_data
GRAPH_FOLDER = "graphs"

os.makedirs(GRAPH_FOLDER, exist_ok=True)


def create_dataframe():
    df = get_training_data()
    # data = {

    #     "product_name":[

    #         "Apple smartphone",

    #         "Samsung Phone",

    #         "Dell Laptop",

    #         "Nike Shoes",

    #         "Adidas Shoes",

    #         "Office Chair",

    #         "Dining Table",

    #         "Microwave",

    #         "Refrigerator",

    #         "Vacuum Cleaner"
    #     ],

    #     "category":[

    #         "Electronics",

    #         "Electronics",

    #         "Electronics",

    #         "Footwear",

    #         "Footwear",

    #         "Furniture",

    #         "Furniture",

    #         "Home Appliances",

    #         "Home Appliances",

    #         "Home Appliances"
    #     ]
    # }
    # return pd.DataFrame(data)
    return df


def generate_count_plot():

    df = create_dataframe()

    plt.figure(figsize=(8,5))

    counts = df["category_name"].value_counts()

    counts.plot(kind="bar")

    plt.title("Category Count")

    plt.xlabel("Category")

    plt.ylabel("Products")

    path = os.path.join(GRAPH_FOLDER,"count_plot.png")

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

    return path

def generate_pie_chart():

    df = create_dataframe()

    counts = df["category_name"].value_counts()

    plt.figure(figsize=(7,7))

    plt.pie(

        counts,

        labels=counts.index,

        autopct="%1.1f%%"

    )

    plt.title("Category Distribution")

    path=os.path.join(

        GRAPH_FOLDER,

        "pie_chart.png"

    )

    plt.savefig(path)

    plt.close()

    return path

def generate_histogram():

    df=create_dataframe()

    df["length"]=df["product_name"].str.len()

    plt.figure(figsize=(8,5))

    plt.hist(df["length"],bins=5)

    plt.title("Product Name Length")

    plt.xlabel("Characters")

    plt.ylabel("Frequency")

    path=os.path.join(

        GRAPH_FOLDER,

        "histogram.png"

    )

    plt.savefig(path)

    plt.close()

    return path