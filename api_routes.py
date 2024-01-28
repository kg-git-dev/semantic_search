from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pickle

api_blueprint = Blueprint('api', __name__)

# Load Sentence Transformer model for semantic search
semantic_search_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load precomputed embeddings, MLS values, and encoded passages from pickle file
with open('embeddings.pkl', 'rb') as file:
    data = pickle.load(file)

corpus_embeddings = data['embeddings']
mls_values = data['mls_values']
encoded_passages = data['encoded_passages']  # Use 'encoded_passages' directly from 'data'

@api_blueprint.route('/query', methods=['POST'])
def process_query():
    try:
        # Get the query from the JSON payload
        query_data = request.get_json()
        user_query = query_data['query']

        # Encode the user query
        query_embedding = semantic_search_model.encode(user_query, convert_to_tensor=True)

        # Perform semantic search
        hits = util.semantic_search(query_embedding.unsqueeze(0), corpus_embeddings, top_k=10)
        hits = hits[0]

        # Prepare the response
        response_data = {
            'user_query': user_query,
            'results': [
                {
                    'passage': ' '.join(encoded_passages[hit['corpus_id']].split('. ')),
                    'mls_value': mls_values[hit['corpus_id']],
                    'score': hit['score']
                }
                for hit in hits
            ]
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
