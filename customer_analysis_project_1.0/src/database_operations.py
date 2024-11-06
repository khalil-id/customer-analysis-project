import pandas as pd
from mysql.connector import Error
from src.database_utils import create_db_connection, execute_query
from src.config import DATA_PATH

def insert_customer_data(customer_data):
    connection = create_db_connection()

    if connection:
        try:
            # Prepare queries with data
            queries = [
                f"INSERT INTO attrition (customer_id, attrition) VALUES ('{customer_data['customer_id']}', {customer_data['prediction']})",
                f"INSERT INTO clients (customer_id, gender, age, estimated_salary, credit_score, products_number, country) VALUES ('{customer_data['customer_id']}', '{customer_data['gender']}', {customer_data['age']}, {customer_data['estimated_salary']}, {customer_data['credit_score']}, {customer_data['products_number']}, '{customer_data['country']}')",
                f"INSERT INTO customers (customer_id, balance, country) VALUES ('{customer_data['customer_id']}', {customer_data['balance']}, '{customer_data['country']}')",
                f"INSERT INTO membership (customer_id, tenure, credit_card, active_member) VALUES ('{customer_data['customer_id']}', {customer_data['tenure']}, {customer_data['credit_card']}, {customer_data['active_member']})"
            ]

            # Execute queries
            for query in queries:
                execute_query(connection, query)

            connection.commit()
            print(f"Data inserted successfully to all tables")
        except Error as e:
            print(f"Error: '{e}'")
        finally:
            if connection.is_connected():
                connection.close()
                print("MySQL connection is closed")
    else:
        print("Error: Could not connect to the database")

# You may want to add more functions here for other database operations