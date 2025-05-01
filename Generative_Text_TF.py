import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import get_file 
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow import linalg
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from document_parser import Parser 
import os
import pandas as pd
import numpy as np


path = "/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/texts"

folder_files = []
for fname in os.listdir(path):
    folder_files.append(os.path.join(path, fname))

data_dict = []
for item, path in enumerate(folder_files):
    parser = Parser(path)
    data = parser.parse_txt()
    data_dict.append(parser.convert_to_json())

data_lines1 = [value for key, value in data_dict[0].items()] # apology 665
data_lines2 = [value for key, value in data_dict[1].items()] # laws 7441
data_lines3 = [value for key, value in data_dict[2].items()] # symposium 1091
data_lines4 = [value for key, value in data_dict[3].items()] # republic 8557
data_lines5 = [value for key, value in data_dict[4].items()] # phaedrus 1253

data_lines_total = data_lines1 + data_lines2 + data_lines3 + data_lines4 + data_lines5 # 19007
len(data_lines_total)

# total word count
word_counter = {} 
for string in data_lines_total: 
     value_list = string.split()
     for i in value_list:
        if i not in word_counter:
            word_counter[i] = 1
        else: word_counter[i] += 1

len(word_counter) #16,525 - total unique words

word_list = []
count_list = []
for key, value in word_counter.items():
    if key not in word_list:
        word_list.append(key) 
        count_list.append(value)

len(word_list) == len(count_list) #16525



TFIDF_data = pd.DataFrame(count_list, index = word_list, columns=["Total Counts"])

for i in word_counter.keys():
    if i not in ["words", "Total Word Count"]:
        TFIDF_data[i] = word_counter[i]

# ['\ufeffthe', 'project', 'gutenberg', 'ebook', 'of', 'apology', 'this', 'is', 'for', 'the']
#[2, 172, 82, 36, 25543, 19, 2592, 10618, 3645, 38681]

# word count by document data_lines1-5
# total word count - 16525
list = np.repeat([0], 16525)
word_counter = {"words": word_list, 
                "Count in Apology": list.copy(), "Count in Laws": list.copy(),"Count in Symposium": list.copy(), 
                "Count in Republic": list.copy(), "Count in Phaedrus": list.copy(), 
                "Total Word Count": list.copy(), 
                "TF Score": list.copy().astype("float"),
                "Docs Containing Word": list.copy(), 
                "IDF Score": list.copy().astype("float"),
                "TF-IDF Score": list.copy().astype("float")}

for idx, value in enumerate(word_counter["words"]):
    for string in data_lines1:
        value_list = string.split()
        for word in value_list:
            if value == word:
                word_counter["Count in Apology"][idx] += 1
                     

for idx, value in enumerate(word_counter["words"]):
    for string in data_lines2:
        value_list = string.split()
        for word in value_list:
            if word == value:
                word_counter["Count in Laws"][idx] += 1       

for idx, value in enumerate(word_counter["words"]):
    for string in data_lines3:
        value_list = string.split()
        for word in value_list:
            if word == value:
                word_counter["Count in Symposium"][idx] += 1

for idx, value in enumerate(word_counter["words"]):
    for string in data_lines4:
        value_list = string.split()
        for word in value_list:
            if word == value:
                word_counter["Count in Republic"][idx] += 1

for idx, value in enumerate(word_counter["words"]):
    for string in data_lines5:
        value_list = string.split()
        for word in value_list:
            if word == value:
                word_counter["Count in Phaedrus"][idx] += 1


# docs defined as sentences
for idx, value in enumerate(word_counter["words"]):
    for i in data_lines_total:
        if value in i:
            word_counter["Docs Containing Word"][idx] +=1


# calculate TF-IDF vectors

# term frequency calculation
# 5 docs are considered the corpus here: 
# Iterate through each one to capture total times each term appears
# Validate the totals added from indiv docs with initial word counts
for key in word_counter.keys():
    if key not in ["words", "Total Word Count", "TF Score", 
                   "Docs Containing Word", "IDF Score"]:
        for idx, value in enumerate(word_counter[key]):
                word_counter["Total Word Count"][idx] += value

# total number of terms in document
total_terms = len(word_counter["words"]) #16,525


# TF score formula
# word count / count of all words by corpus
for idx, value in enumerate(word_counter["words"]):
    word_counter["TF Score"][idx] = word_counter["Total Word Count"][idx] / total_terms


# IDF score
total_docs = 19007 #number of strings in 5 books

# implement IDF formula: 
# total docs / total docs containing words
for idx, value in enumerate(word_counter["words"]):
    if word_counter["Docs Containing Word"][idx] != 0:
        word_counter["IDF Score"][idx] = total_docs/word_counter["Docs Containing Word"][idx]
    elif word_counter["Docs Containing Word"][idx] == 0 :
        print(idx)

        

for idx, value in enumerate(word_counter["words"]):
    word_counter["TF-IDF Score"][idx] = word_counter["TF Score"][idx] * np.log(word_counter["IDF Score"][idx])


#---------------- inspect scores
for keys in word_counter.keys():
    print(f"key: {keys}, {word_counter[keys][2158]}") 

#51:  laws,      0.1638, 
#825: children,  0.09448 
#52:  country,   0.06427
#843: justice,   0.11239
#2158: family,   0.04405

word_counter["words"].index("family")

#------------------------ Vectorize the text data: converts vocabulary to unique int identifiers
vocab_size = 16526
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="tf_idf", vocabulary=word_counter["words"], idf_weights=word_counter["TF-IDF Score"]) #create vocab layer
#text_ds = tf.data.Dataset.from_tensor_slices(data_lines).batch(100)
vectorizer.adapt(data_lines_total) #determines frequency of indiv string values, creates vocabulary from strings

vectorized_text = vectorizer(data_lines_total)[0] #map integers to learned embeddings
print("Vectorized text shape:", vectorized_text.shape) #Vectorized text shape: (16526)
print("First 10 vectorized tokens:", vectorized_text.numpy()[:10]) 

vectorizer.vocabulary_size() #16526
vectorizer.get_vocabulary()

# Create sequences for X, Y
def create_sequences(text, seq_length): 
    input_seqs = [] 
    target_seqs = [] 
    for i in range(len(text) - seq_length): 
        input_seq = text[i:i + seq_length] 
        target_seq = text[i + 1:i + seq_length + 1] 
        input_seqs.append(input_seq) 
        target_seqs.append(target_seq) 
    return np.array(input_seqs), np.array(target_seqs) 

X, Y = create_sequences(vectorized_text.numpy(), seq_length=50) # (16526, 50)

print("Number of sequences generated:", len(X)) #16476 sequences generated
print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated") 

# Check if X and Y are not empty 
assert X.size > 0, "Input data X is empty" 
assert Y.size > 0, "Target data Y is empty" 

X = tf.convert_to_tensor(X) 
Y = tf.convert_to_tensor(Y) 
print("Shape of X:", X.shape) #Shape of X: (16,476, 50)
print("Shape of Y:", Y.shape) #Shape of Y: (16,476, 50)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8): #embed_dim also meaning key_dim
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TransformerModel(Model):  # Model is now properly imported
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embed = embed_dim
        self.embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)
    
    def create_causal_mask(self, seq_length):
        """Creates a causal mask for self-attention.
        The mask prevents the model from attending to future positions."""
        # Create a lower triangular matrix (1s in the lower triangle, 0s elsewhere)
        mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        # Convert to proper dtype and reshape for broadcasting
        mask = tf.cast(mask, dtype=tf.float32)
        # The attention mechanism uses 1 for masked positions and 0 for valid positions
        # # So we need to convert our mask (swap 0s and 1s and multiply by a large negative)
        mask = mask * -1e9
        return mask
    
    def positional_encoding(self, seq_length, embed_dim):
        positions = np.arange(seq_length)[:, np.newaxis]
        depths = np.arange(embed_dim)[np.newaxis, :]#
        angle_rads = self.get_angles(positions, depths, embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, depths, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(embed_dim))
        return pos * angle_rates
    
    def call(self, inputs, training=False, mask=True):
        length = tf.shape(inputs)[1] 
        causal_mask = self.create_causal_mask(length)
        x = self.embedding(inputs)#
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embed, tf.float32))
        x += self.pos_encoding[:, :length, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=causal_mask)  # Pass training argument correctly
            output = self.dense(x)
        return output
    
# Hyperparameters 
embed_dim = 512
num_heads = 8 #common value, to use 64 dimensions for each head
ff_dim = 2048 #standard is 4x embed_dim
num_layers = 10
dropout_rate = 0.2 

# Build the Transformer model 
model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length=50)

# Provide input shape to build the model by passing a dummy input with maxval specified
_ = model(tf.random.uniform((1, 50), maxval=vocab_size, dtype=tf.int32))

# Compile the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Summary of the model 
model.summary()

# Early stopping callback to stop training if the loss doesn't improve
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Train the transformer model on the full input and target sequences
history = model.fit(X, Y, epochs=2, batch_size=32, callbacks=[early_stopping])

# Plot training loss to monitor model performance over epochs
plt.plot(history.history['loss']) #.0028
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()





# Save/load the model weights-----------------------------
model.save_weights('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/transformer_model.weights.h5')
# to load the model weights, compile the model first, then load the weights
# Rebuild the model with the same architecture
model.load_weights('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/transformer_model.weights.h5')

# Save model
model.save('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/transformer_model.keras')
new_model = tf.keras.models.load_model('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/transformer_model.keras')
new_model.summary()


# Extract embedding weight values and save to disk--------
import io
weights = model.embedding.get_weights()[0]
vocab = vectorizer.get_vocabulary()

# write to disk as csv
weights = pd.DataFrame(weights)
weights["vocab"] = vocab
weights.to_csv('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/embedding/embeddings_tfidf.csv')


# write to disk as tsv
out_v = io.open('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/embedding/vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('/Users/tdubon/Documents/LaptopFiles/Projects/PlatoAgent/embedding/metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")

# Export the vectors and metadata to lists for visualization
vectors = []
words = []
for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  vectors.append('\t'.join([str(x) for x in vec]) + "\n")
  words.append(word)  

out_v.close()
out_m.close()

#-----------------------
# modifying model layers when weights are imported 
model.weights[0].value
model.layers[0].set_weights(embedding_matrix)
model.layers[0].trainable = False #freeze embedding layer
weights = model.embedding.get_weights()[0]

# save checkpoints during training-----------------------------------
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#As long as two models share the same architecture you can share weights between them.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(X, Y,  
          epochs=10,
          #validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


#---------------------- Fit model to new text


# Convert the data_dict to a list of strings for vectorizer
data_lines = []
for key, value in data_dict.items():
    data_lines.append(value)
data_lines = data_lines[3:] 

# Vectorize the text data
vocab_size = 10350
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int")
text_ds = tf.data.Dataset.from_tensor_slices(data_lines).batch(7570)
vectorizer.adapt(text_ds)

vectorized_text = vectorizer(data_lines)[0]
print("Vectorized text shape:", vectorized_text.shape) 
print("First 10 vectorized tokens:", vectorized_text.numpy()[:10]) 

X, Y = create_sequences(vectorized_text.numpy(), seq_length=50) 

print("Number of sequences generated:", len(X)) 
print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated") 

# Check if X and Y are not empty 
assert X.size > 0, "Input data X is empty" 
assert Y.size > 0, "Target data Y is empty" 


X = tf.convert_to_tensor(X) 
Y = tf.convert_to_tensor(Y) 
print("Shape of X:", X.shape) 
print("Shape of Y:", Y.shape) 


# Train the transformer model on the full input and target sequences
history = model.fit(X, Y, epochs=20, batch_size=32, callbacks=[early_stopping])




# --------------------------Generate text using the trained model
seq_length=50

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    # Convert the start string to a vectorized format
    input_eval = vectorizer([start_string]).numpy()
    
    # Ensure the input length is the same as the model's expected input shape
    if input_eval.shape[1] < seq_length:
        # Pad the input if it's shorter than the expected sequence length
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        # Truncate the input if it's longer than the expected sequence length
        input_eval = input_eval[:, -seq_length:]
    
    input_eval = tf.convert_to_tensor(input_eval)
    
    # Initialize an empty list to store generated text
    text_generated = []
    
    # Start generating text
    for i in range(num_generate):
        # Make predictions using the model
        predictions = model(input_eval)
        
        # Remove only the batch dimension, keep the logits as 2D (batch_size, vocab_size)
        predictions = predictions[0]  # This should be of shape [vocab_size]
        
        # Apply temperature to predictions
        predictions = predictions / temperature
        
        # Use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        
        # Update the input tensor to include the predicted word, maintaining the sequence length
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)  # Append predicted token
        input_eval = input_eval[:, -seq_length:]  # Keep only the last `seq_length` tokens
        input_eval = tf.convert_to_tensor(input_eval)  # Convert back to tensor
        
        # Append the predicted word to the generated text
        text_generated.append(vectorizer.get_vocabulary()[predicted_id])
        
    # Return the generated text starting from the initial seed
    return start_string + ' ' + ' '.join(text_generated)

# Generate text with temperature control
# Lower temperature for more focused predictions
start_string = "War is not in the intrinsic nature of man. Explain why you think it is. Your response doesn't make sense. Try again."
generated_text = generate_text(model, start_string, num_generate=15, temperature=4)  
print(generated_text)

prompts = ["What is the purpose of implementing a system of laws? ",
 "What is the purpose of a system of laws? ",
 "What is the purpose of a system of government? ",
 "What is the purpose of a system of education? ",
 "describe a system of ethics ",
 "what is unethical ",
 "What is the purpose of a system of justice? ",
 "What is the purpose of a system of punishment? ",
 "What is the purpose of a system of reward? ",
 "What is the purpose of a system of morality? ",
 "What is the purpose of a system of ethics? ",
 "What is the purpose of a system of philosophy? ",
 "what is a conscience? ",
 "What is the purpose of science? ",
 "What is the purpose of a system of religion? ",
 "Describe human nature",
 "Describe the nature of mankind",
 "List 2 of the most important concepts that govern human nature.",
 "Elaborate on 3 points that are the most important topics in the Republic.",
 "What are the fundamental principles of equality?",
 "What is the nature of freedom?",
 "How does man relate to nature? ",
 "What is man's responsibility to nature? ",
 "What is man's responsibility to his neighbor? ",
 "What is man's responsibility to his family?",
 "What is the role of a woman in society? ",
 "Persuasively explian why peace is achievable without savagery",
 "Persuasively explain why peace is achievable without violence"]

# word search
word1 = "family"
word1_vector = []
word1_quotes = []
word2 = "justice"
word2_vector = []
word2_quotes = []

for indx, line in enumerate(data_lines_total):
    if word1 in line:
        word1_quotes.append(line)
    if word2 in line:
        word2_quotes.append(line)

# vector search
for i, w in enumerate(weights["vocab"]):
    if w == word1:
        word1_vector.append(weights.loc[i, 0:499])
    elif w == word2:
        word2_vector.append(weights.loc[i, 0:499])

# plotting values for each vector
fig, ax = plt.subplots()
scatter = plt.plot(word1_vector[0][499], 1, color='green')
scatter = plt.plot(word2_vector[0][499], 1, color='purple')


x_increments = np.arange(-.05, .05, .0025)
y_increments = np.arange(0, 1.5, .5)
plt.xticks(x_increments)
plt.yticks(y_increments)
plt.xlabel('Embedding Values')
#plt.ylabel()
plt.title('Embeddings for Family and Justice')
plt.grid(True)
plt.show()
plt.close()
   
# distance between vectors

plt.gca().arrow(0, 0, word1_vector[0][499], word2_vector[0][499],fc="green")
