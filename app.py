from flask import Flask, request, jsonify, render_template
import re
import numpy as np

app = Flask(__name__)

def preprocess_text(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def sentence_similarity(sent1, sent2):
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    common_words = words1.intersection(words2)
    
    if len(words1) == 0 or len(words2) == 0:
        return 0
    
    return len(common_words) / (len(words1) + len(words2))

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return similarity_matrix

def pagerank(matrix, eps=0.0001, d=0.85):
    n = matrix.shape[0]
    ranks = np.ones(n) / n
    delta = 1
    
    while delta > eps:
        new_ranks = (1 - d) / n + d * matrix.T.dot(ranks)
        delta = np.linalg.norm(new_ranks - ranks)
        ranks = new_ranks
    
    return ranks

def summarize_text(text, num_sentences=3):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    ranks = pagerank(similarity_matrix)
    
    ranked_sentences = [sentence for rank, sentence in sorted(zip(ranks, sentences), reverse=True)]
    
    summary = ' '.join(ranked_sentences[:num_sentences])
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.form['text']
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    num_sentences = int(request.form['numSentences'])
    summary = summarize_text(data, num_sentences)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=False)
