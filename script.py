import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import PyPDF2

class DocumentSimilarityMatcher:
    def __init__(self, train_path):
        self.train_path = os.path.abspath(train_path)
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"The training directory does not exist: {self.train_path}")
        if not os.path.isdir(self.train_path):
            raise NotADirectoryError(f"The specified path is not a directory: {self.train_path}")

        self.train_data = self.load_data(self.train_path)
        if not self.train_data:
            raise ValueError("No valid data found in the training directory")
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.train_data.values())
        except ValueError as e:
            print("Error in vectorization:")
            print(f"Number of documents: {len(self.train_data)}")
            print(f"Sample document content: {next(iter(self.train_data.values()))}")
            raise e

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip()

    def load_data(self, path):
        data = {}
        try:
            for filename in os.listdir(path):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(path, filename)
                    try:
                        content = self.extract_text_from_pdf(file_path)
                        if content:  # Only add non-empty content
                            data[filename] = content
                        else:
                            print(f"Warning: Empty content for file {filename}")
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
        except Exception as e:
            print(f"Error accessing directory {path}: {str(e)}")
        
        if not data:
            print(f"Warning: No valid data found in the specified directory: {path}")
        return data

    def find_similar_documents(self, query_doc, top_n=5):
        query_vector = self.vectorizer.transform([query_doc])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        
        return [(list(self.train_data.keys())[i], cosine_similarities[i]) for i in related_docs_indices]

def process_test_data(matcher, test_path):
    test_path = os.path.abspath(test_path)
    if not os.path.exists(test_path):
        print(f"Error: The test directory does not exist: {test_path}")
        return {}
    if not os.path.isdir(test_path):
        print(f"Error: The specified test path is not a directory: {test_path}")
        return {}

    test_data = matcher.load_data(test_path)
    if not test_data:
        print("No valid test data found")
        return {}

    results = {}
    for filename, content in test_data.items():
        similar_docs = matcher.find_similar_documents(content)
        results[filename] = similar_docs
    return results

def main():
    # Use raw string for Windows paths
    train_path = r'.\sample_invoices\train'
    test_path = r'.\sample_invoices\test'

    try:
        # Initialize the matcher with the training data
        matcher = DocumentSimilarityMatcher(train_path)

        # Process test data
        results = process_test_data(matcher, test_path)

        # Print results
        for test_file, matches in results.items():
            print(f"\nTest Case: {test_file}")
            for doc, score in matches:
                print(f"Matched: {doc}, Similarity Score: {score:.4f}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()