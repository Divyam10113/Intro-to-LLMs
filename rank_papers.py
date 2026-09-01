import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress some noisy warnings from huggingface
warnings.filterwarnings('ignore')

def rank_papers():
    # 1. Load papers
    print("Loading llm_healthcare_papers.csv...")
    df = pd.read_csv('llm_healthcare_papers.csv')
    
    # 2. Load embedding model
    print("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Define the topic
    topic = "Generalizable vision-language models for zero-shot/few-shot anomaly detection in medical images"
    
    # Encode the abstracts and the topic
    print("Encoding abstracts and topic...")
    abstracts = df['Abstract'].fillna('').tolist()
    abstract_embeddings = model.encode(abstracts)
    topic_embedding = model.encode([topic])
    
    # 4. Compute cosine similarity
    print("Computing cosine similarity...")
    similarities = cosine_similarity(topic_embedding, abstract_embeddings)[0]
    
    df['Similarity'] = similarities
    
    # 5. Sort by similarity
    df_sorted = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    
    # Save the ranked results
    output_file = 'ranked_papers.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\nSaved ranked papers to {output_file}\n")
    
    # Print the top 20 for manual inspection
    print("="*60)
    print("TOP 20 PAPERS FOR MANUAL INSPECTION")
    print("="*60)
    
    top_20 = df_sorted.head(20)
    for i, row in top_20.iterrows():
        print(f"[{i+1}] Score: {row['Similarity']:.4f}")
        print(f"Title: {row['Title']}")
        print(f"Abstract: {row['Abstract']}")
        print("-" * 60)

if __name__ == "__main__":
    rank_papers()
