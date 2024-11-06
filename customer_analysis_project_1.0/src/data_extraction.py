import pandas as pd
from src.database_utils import create_db_connection, execute_query
from src.config import DATA_PATH

def extract_and_save_data():
    connection = create_db_connection()
    if connection:
        query = """
        SELECT 
            c.customer_id,
            c.gender,
            c.age,
            c.estimated_salary,
            c.credit_score,
            c.products_number,
            m.tenure,
            m.credit_card,
            m.active_member,
            a.attrition,
            co.balance,
            c.country
        FROM 
            clients c
        JOIN 
            membership m ON c.customer_id = m.customer_id
        JOIN 
            attrition a ON c.customer_id = a.customer_id
        JOIN 
            comptes co ON c.customer_id = co.customer_id
        """
        
        results = execute_query(connection, query)
        connection.close()
        
        if results:
            columns = ['customer_id', 'gender', 'age', 'estimated_salary', 'credit_score', 
                       'products_number', 'tenure', 'credit_card', 'active_member', 'attrition', 
                       'balance','country']
            df = pd.DataFrame(results, columns=columns)
            df.to_csv(DATA_PATH, index=False)
            print(f"Data exported successfully to {DATA_PATH}")
            return df
        else:
            print("No data extracted.")
            return None