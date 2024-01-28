from sentence_transformers import SentenceTransformer
import sqlite3
import pickle

# Load Sentence Transformer model for semantic search
semantic_search_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to SQLite database
db_connection = sqlite3.connect('residentialDatabase.db')
db_cursor = db_connection.cursor()

# Execute query to fetch data
db_cursor.execute("SELECT * FROM residentialDatabase WHERE MLS IS NOT NULL")
rows = db_cursor.fetchall()

# Get column names from the cursor description
column_names = [desc[0] for desc in db_cursor.description]

# Initialize arrays to store encoded passages and MLS values
encoded_passages = []
mls_values = []

# Encode property details
for row in rows:
    passage = ""
    mls_value = None

    for i, value in enumerate(row):
        column_name = column_names[i]
        if value is not None:
            passage += f"{column_name}: {value}. "
            if column_name == 'MLS':
                mls_value = value

    encoded_passages.append(passage)
    mls_values.append(mls_value)

# Encode all passages
corpus_embeddings = semantic_search_model.encode(encoded_passages)

# Save the embeddings, MLS values, and encoded passages to a pickle file
with open('embeddings.pkl', 'wb') as file:
    pickle.dump({'embeddings': corpus_embeddings, 'mls_values': mls_values, 'encoded_passages': encoded_passages}, file)
