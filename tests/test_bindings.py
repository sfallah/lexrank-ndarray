import lexrank_ndarray
import numpy as np

def test_lexrank_py():
    """
    Test the lexrank_py function from the lexrank_ndarray module.
    Computes LexRank scores for sample embeddings.
    """

    # Sample embedding
    num_sentences = 10
    embedding_dim = 300  
    np.random.seed(42)  # For reproducibility
    embeds = np.random.rand(num_sentences, embedding_dim).astype(np.float32).tolist()

    # Parameters for LexRank
    max_iter = 10000     
    threshold = 0.1

    try:
        # Call the lexrank_py function from the lexrank_ndarray module
        ranked_sentences = lexrank_ndarray.lexrank_py(embeds, max_iter, threshold)

        # Sort the sentences by their LexRank scores in descending order
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)

        # Display the ranked sentences
        print("Ranked Sentences:")
        for rank, (sentence_idx, score) in enumerate(ranked_sentences, start=1):
            print(f"{rank}. Sentence {sentence_idx} - Score: {score:.6f}")

    except Exception as e:
        print(f"An error occurred: {e}")

def test_cosine_similarity_py():
    """
    Test the cos_similarity_py function from the lexrank_ndarray module.
    Computes cosine similarity between two sample embeddings.
    """
    # Generate two sample embeddings
    embedding1 = np.random.rand(300).astype(np.float32).tolist()
    embedding2 = np.random.rand(300).astype(np.float32).tolist()

    print("\nTesting cos_similarity_py with two random embeddings...")
    try:
        # Call the cos_similarity_py function
        similarity = lexrank_ndarray.cos_similarity_py(embedding1, embedding2)

        # Display the cosine similarity
        print(f"Cosine Similarity: {similarity:.6f}")

    except Exception as e:
        print(f"An error occurred during cos_similarity_py test: {e}")


def main():
    """
    Main function to execute all tests.
    """
    test_lexrank_py()
    test_cosine_similarity_py()

if __name__ == "__main__":
    main()
