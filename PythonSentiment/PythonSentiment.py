import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding


#ucitavamo podatke
df = pd.read_csv("C:\\Users\\katarina.stanojkovic\\Downloads\\archive (3)\\Tweets.csv")
#uzimamo samo tekst i ocenu
review_df = df[['text','airline_sentiment']]
print(review_df)
review_df = review_df[review_df['airline_sentiment'] != 'neutral']
#niz ocena prevodimo u numericki niz
sentiment_label = review_df.airline_sentiment.factorize()

#izvlacimo samo komentare
tweet = review_df.text.values
print(tweet)


tokenizer = Tokenizer(num_words=10000)
#mapiranje reci na brojeve
tokenizer.fit_on_texts(tweet)

#zamenjuje reci brojevima
encoded_docs = tokenizer.texts_to_sequences(tweet)

#da sve recenice imaju istu duzinu
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


embedding_vector_length = 32
#kreiranje sekvencijalnog modela, dodajemo mu slojeve
model = Sequential()
#ovaj sloj mapira reci na vektore, recnik sadrzi 10000 reci, definisana je duzina vektora i ocekuje se ulaz od 200 karaktera
model.add(Embedding(10000, embedding_vector_length, input_length=200))
#dropout mehanizam, 25% neurona iz vektora se slucajno odbacuju
model.add(SpatialDropout1D(0.25))
#neuronska mreza sa 50 neurona
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
#iskljucuje 20% izlaza 
model.add(Dropout(0.2))
#sloj gustine?
model.add(Dense(1, activation='sigmoid'))
#postavlja se gubitak, optimizer i metrika po kojoj se vrsi evaluacija
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


test_sentence1 = "I enjoyed my journey on this flight."
predict_sentiment(test_sentence1)

test_sentence2 = "This is the worst flight experience of my life!"
predict_sentiment(test_sentence2)



