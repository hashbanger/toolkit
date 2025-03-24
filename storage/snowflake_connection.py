import os
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

class SnowflakeConnector:
    """
    A class used to manage the connection to Snowflake, fetch data, transpose it, and save it.

    Attributes:
        user_id (str): User ID for Snowflake connection.
        password (str): Password for Snowflake connection.
        engine (Engine): SQLAlchemy Engine object for Snowflake.
    """

    def __init__(self, account: str, warehouse: str, database: str, schema: str):
        """
        Initializes the SnowflakeConnector class, loads environment variables and constants, sets up the Snowflake connection, and initializes AWS S3 utilities.
        """
        load_dotenv()
        self.user_id = os.getenv('SNOWFLAKE_USER_ID')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.engine = self.get_snowflake_engine()


    def get_snowflake_engine(self) -> Any:
        """Establishes and returns a connection to Snowflake"""
        if not self.user_id or not self.password:
            raise Exception("Unable to find snowflake credentials. SNOWFLAKE_USER_ID and SNOWFLAKE_PASSWORD must be provided in the .env file.")
        
        engine = create_engine(
            URL(
                user=self.user_id,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema # TODO: CHECK IF APPLICABLE
                # account='od08276.us-east-2.aws',
                # warehouse='BCG_VW',
                # database="REVELIO",
                # schema='BCG_READER'
                # account='ua02898.us-east-2.aws',
                # warehouse='BCG_POP',
                # database="BCG_POP_INBOUND",
                # schema='BCG'
            )
        )
        return engine

    def execute_query(self, query: str, return_as_dataframe: bool=True) -> Optional[List]:
        """Executes a single SQL query on the Snowflake database and returns the results as a list of tuples"""
        try:
            # Assuming self.engine is a SQLAlchemy engine or similar
            with self.engine.connect() as connection:
                if return_as_dataframe:
                    try:
                        df = pd.read_sql(query, connection)
                        return df
                    except Exception as e:
                        print(f"Failed to execute query: {e}")
                        return pd.DataFrame()
                else:
                    with connection.begin():
                        result_proxy = connection.execute(query)
                        results = None

                        # Check if the query returns a result set
                        if result_proxy.returns_rows:
                            results = result_proxy.fetchall()
                            print("Query executed successfully with results.")
                        else:
                            print("Query executed successfully without results.")

                        return results
            
        except Exception as e:
            print("Failed to execute query:", e)
            raise

    def execute_batch(self, query: str, data: List[Tuple]):
        """Executes a batch SQL query on the Snowflake database using executemany"""
        try:
            with self.engine.connect() as connection:
                with connection.begin():
                    connection.execute(query, data)
                    print("Batch query executed successfully.")
        except Exception as e:
            print("Failed to execute batch query:", e)
            raise