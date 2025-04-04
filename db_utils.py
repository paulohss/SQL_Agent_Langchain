from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from util.logger import log 

# Create engine and session
engine = create_engine('sqlite:///example.db')
SessionLocal = sessionmaker(bind=engine)

# ------------------------------------------------
# Get database schema
# ------------------------------------------------
def get_database_schema():
    inspector = inspect(engine)
    schema = ""
    
    for table_name in inspector.get_table_names():
        schema += f"Table: {table_name}\n"
        for column in inspector.get_columns(table_name):
            col_name = column["name"]
            col_type = str(column["type"])
            if column.get("primary_key"):
                col_type += ", Primary Key"
            if column.get("foreign_keys"):
                fk = list(column["foreign_keys"])[0]
                col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
            schema += f"- {col_name}: {col_type}\n"
        schema += "\n"
    
    log.info("Retrieved database schema.")
    
    return schema