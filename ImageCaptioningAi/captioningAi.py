import string
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
vgg_model = VGG16(weights='imagenet', include_top=False)

def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip header or empty lines
            if not line or line.startswith("image_caption"):
                continue
            # Split the line based on the comma separator
            image_id, caption = line.split(',', 1)
            image_id = image_id.strip()
            caption = caption.strip()

            # Store the caption with the image ID as the key
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions


def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in captions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

def create_tokenizer(captions):
    lines = []
    for key in captions.keys():
        for desc in captions[key]:
            lines.append(desc)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(captions):
    return max(len(desc.split()) for key in captions.keys() for desc in captions[key])

def extract_image_features(directory):
    
    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature.flatten()
    return features
def create_sequences(tokenizer, max_length, captions, features, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in captions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(features[key])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_captioning_model(vocab_size, max_length):
    # Feature extractor (from VGG16)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model (LSTM)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder model (merging VGG and LSTM)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
# Load captions
filename = 'ImageCaptioningAi/archive/captions.txt'
captions = load_captions(filename)
clean_captions(captions)

# Prepare tokenizer
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1
print(f"Number of captions loaded: {len(captions)}")
max_len = max_length(captions)

# Extract image features
image_directory = 'ImageCaptioningAi/archive/Images'
features = extract_image_features(image_directory)

# Prepare sequences
X1, X2, y = create_sequences(tokenizer, max_len, captions, features, vocab_size)

# Define the model
captioning_model = define_captioning_model(vocab_size, max_len)

# Train the model
captioning_model.fit([X1, X2], y, epochs=20, verbose=2)

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load a new image and extract features
image_path = 'black.jpg'
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
new_features = vgg_model.predict(image, verbose=0).flatten().reshape((1, 4096))

# Generate caption
caption = generate_caption(captioning_model, tokenizer, new_features, max_len)
print("Starting caption cleaning...")
clean_captions(captions)
print("Finished caption cleaning.")

print("Starting tokenizer creation...")
tokenizer = create_tokenizer(captions)
print("Finished tokenizer creation.")

print("Starting image feature extraction...")
features = extract_image_features(image_directory)
print("Finished image feature extraction.")

print("Generated Caption:", caption)

