
import numpy as np

class VectorDatabase:

    def __init__(self):
        self.vectors = []

    def add_vector(self, vec_id, vector, metadata=None):
        record = {
            "id": vec_id,
            "vector": np.array(vector, dtype=np.float32),
            "metadata": metadata
        }

        self.vectors.append(record)

    def get_all_vectors(self):
        return self.vectors

    def _cosine_similarity(self, vec_a, vec_b):

        dot_product = np.dot(vec_a, vec_b)

        norm_a = np.linalg.norm(vec_a)

        norm_b = np.linalg.norm(vec_b)

        cos_sim = dot_product / (norm_a * norm_b + 1e-8)

        return cos_sim

    def search(self, query_vector, top_k=3):

        query_vector = np.array(query_vector, dtype=np.float32)

        results = []

        for record in self.vectors:
            sim = self._cosine_similarity(query_vector, record['vector'])

            results.append({
                'id': record['id'],
                'similarity': sim,
                'metadata': record['metadata']
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

db = VectorDatabase()

db.add_vector('vec1', np.random.rand(5), metadata= { 'text': 'First item' })
db.add_vector('vec2', np.random.rand(5), metadata= { 'text': 'Second item' })
db.add_vector('vec3', np.random.rand(5), metadata= { 'text': 'Third item' })
db.add_vector('vec4', np.random.rand(5), metadata= { 'text': 'Fourth item' })
db.add_vector('vec5', np.random.rand(5), metadata= { 'text': 'Fifth item' })
        
query_vector = np.random.rand(5)

