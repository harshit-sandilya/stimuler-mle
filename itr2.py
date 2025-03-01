import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier

grammar_examples = json.load(open("sample_data/grammar.json"))
vocab_examples = json.load(open("sample_data/vocab.json"))

print(f"Grammar Examples: {len(grammar_examples)}, Vocabulary Examples: {len(vocab_examples)}")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = embedding_model.tokenizer

def get_sentence_embedding(text):
    """Convert sentence to its embedding using SentenceTransformer."""
    return embedding_model.encode(text, convert_to_tensor=True)

def get_index(text,word):
    words = text.split()
    error_words = word.split()

    for i in range(len(words) - len(error_words) + 1):
        if words[i:i + len(error_words)] == error_words:
            return list(range(i + 1, i + len(error_words) + 1))

    return [-1]

def get_vocab_index(word):
    word = word.split()[0]
    return tokenizer.convert_tokens_to_ids(word)

# grammar model
X = [get_sentence_embedding(example['sentence']) for example in grammar_examples]
Y = [get_index(example['sentence'],example['errors'][0])[0] for example in grammar_examples]
X = torch.stack(X)
Y = torch.tensor(Y)

print(X.shape)
print(Y.shape)

x_train = X[:int(len(X) * 0.8)].cpu()
y_train = Y[:int(len(Y) * 0.8)].cpu()
x_test = X[int(len(X) * 0.8):].cpu()
y_test = Y[int(len(Y) * 0.8):].cpu()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# For suggestion Y
print("Suggestions:")
Y = [get_vocab_index(example['improvements'][0]) for example in grammar_examples]
Y = torch.tensor(Y)
print(Y.shape)

x_train = X[:int(len(X) * 0.8)].cpu()
y_train = Y[:int(len(Y) * 0.8)].cpu()
x_test = X[int(len(X) * 0.8):].cpu()
y_test = Y[int(len(Y) * 0.8):].cpu()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
