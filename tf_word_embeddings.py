# The following code inputs txt files from the Internet Archive's Banned Books collection: https://archive.org/details/bannedbooks.
# It outputs word embeddings, a "numeric vector input that represents a word in a lower-dimensional space" (don't ask I don't really understand the math)
# The bottomline is that word embeddings provide a means of grouping words with similar meanings. Word embeddings can be used are the basis of simple stuff 
# like word suggestions in gmail all the way up to crazy new stuff like Chat-GPT. More info can be found at: https://www.geeksforgeeks.org/word-embeddings-in-nlp/.
# In this case, I'm just using it to make some cool visualizations of the lexicon employed in Banned Books
# My work is based off the following guides: https://www.tensorflow.org/text/guide/word_embeddings and https://www.tensorflow.org/tutorials/keras/text_classification

import io
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.layers import TextVectorization

txt_dataset = r'C:\Users\aosgo\OneDrive\Documents\Simmons\Fall 2022\Data Interoperability\Project 3\ML-bannedbooks'

# PART 1 - CREATE DATASET
# Documentation for this part at: https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory
# First we need to assemble the dataset from the directory the text files are saved in
# Not sure exactly what batch_size or seed do, using the same that they used in the word embeddings tutorial
batch_size = 1024
seed = 123
# Validation_split determines what % of the dataset will be used to test the weights developed during training 
train_ds = tf.keras.utils.text_dataset_from_directory(
    txt_dataset, batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
    txt_dataset, batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

# The cache and prefetch methods modify the dataset to help the algorithm run faster, more info at: https://www.tensorflow.org/guide/data_performance 
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Note that I skipped the "using the embedding layer" part of the tutorial because I don't think that feeds into the rest of the code, it's just an explanation of how embedding works

# PART 2 - PREPROCESSING
# Next is text preprocessing, which modifies the text to ready it for the machine learning model. I did a bunch of preprocessing already (i.e. strip().split(' '), 
# deleted extra commas, etc.), but there's still some left, and then we have to vectorize the text files (turn the words into math so we can perform the statistical operations on them
# that go on in an ML model)
# Not sure what to use for vocabulary size and number of words in a sequence. I think the vocab_size should be the total number of different words in the collection. 
# I'm just using 10k because that's what the tutorial did and it seems plenty large enough, but if I want to improve the model I could probably count the dif number of words 
# using pandas and the csv files I made.
vocab_size = 10000
# Sequence length is the one I'm not sure about. I think the sequence length should = the average number of words per book in collection, but the tutorial uses 100 which I used in my
# initial couple runs, and they each took ~45+ minutes to run, so I figured I just don't have the computing power to use longer sequence_lengths.
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to integers.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary. I don't think this is necessary because this dataset is unlabeled, but I'm keeping it for now.
text_ds = train_ds.map(lambda x, y: x) # ***Note*** apparently lambda won't work in tf after 09-2023, so if you're running this after that it's gonna break
vectorize_layer.adapt(text_ds)

# PART 3 - CREATE CLASSIFICATION MODEL
# Next up we use the keras sequential api to define a "continuous bag of words" style model, more on keras' api at https://www.tensorflow.org/guide/keras/sequential_model and
# refer to the "Create a classification model" section of https://www.tensorflow.org/text/guide/word_embeddings for a quick guide on each parameter in the sequential model
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

# PART 4 - COMPILE AND TRAIN MODEL
# TensorBoard visualizes metrics like loss (how "wrong" the model is), this line should open an extension in VSC and then open a new tab that displays loss and accuracy metrics
# Note that accuracy will always = 1 because the data is not labeled, therefore there are no predictions.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

# Then we compile and train the model using the Adam optimizer and BinaryCrossentopy loss. Idk why Adam and BinaryCrossentropy specifically, but every simple ML model I do a 
# tutorial of seems to use those.
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

# Model summary displays basic info about each layer of the model
model.summary()

# PART 5 - RETRIEVE AND SAVE WORD EMBEDDINGS

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

# PART 6 - DISPLAYING THE DATA
# After you've run this file, go to http://projector.tensorflow.org/, click "Load," upload vecs.tsv and meta.tsv, which are the output files created in Part 5
# This should show all the embeddings labeled by word name in vector space. They are grouped by contextual likeness - i.e. the closer they are the more alike they are.
# Note that you can search any of the words in the search bar to the left and see the closest/most alike words to it

