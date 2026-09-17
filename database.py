import mysql.connector
import pandas as pd
from config import (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME,DB_PORT)

def get_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

def get_training_data():

    conn = get_connection()

    query = """
            SELECT 
            p.id, 
            p.product_name, 
            c.category_name, 
            p.brand, 
            p.description, 
            p.price, 
            p.stock, 
            p.total_sales 
            FROM 
            products p 
            INNER JOIN product_categories c ON c.id = p.category_id 
            WHERE 
            p.status = 'Active' 
            ORDER BY 
            p.product_name
    """

    df = pd.read_sql(query, conn)

    conn.close()

    return df