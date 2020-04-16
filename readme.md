
[![General Assembly Logo](https://camo.githubusercontent.com/1a91b05b8f4d44b5bbfb83abac2b0996d8e26c92/687474703a2f2f692e696d6775722e636f6d2f6b6538555354712e706e67)](https://generalassemb.ly/education/web-development-immersive)
![Misk Logo](https://i.ibb.co/KmXhJbm/Webp-net-resizeimage-1.png)

# Speech Emotion Recognition 
Speech is the most natural way of expressing ourselves as humans. Moreover, with the advance technology in self assitance like Siri we can embedding speech emotion recognition (SER) to identify user mode and emotion to response accordingly. Also, SER could help business retain revenue and earn new customer, by measuring their costumer experience. So, it's natural to extand this to a computer application as a collection of methodologies that process and classify speech signals to detect the embedded emotions.
![](https://cdn-images-1.medium.com/freeze/max/1000/0*tqQ-x7QM2zKhJB9F.jpg)
# Problem Statement:
- Customer Experience. 
Since we are different as human being we express emotions differently which results in misunderstanding. One of the challenges to knowing the other’s feeling is by their facial expressions and tone of voice. So, How Call Center or Customer experience know of  their customer satisfaction
- Hearing impairment.
This is easy with people without any disability but with hearing impairment  people detecting emotions may be very challenging even with speech disorder. So, with SER people with hearing impairment can get their message with the emotion which make the communication human-like.
 

# Solutions: 
By using librosa library on audio speech we can extract feature for further analysis, by developing model to analysis speech signal then predict and classify  speech sounds into emotions.

# Audio Data 
- English Audio Speech data [RAVDES](https://zenodo.org/record/1188976#.XpiSSi2B3RY)
- Arabic Audio Speech data [Kaggle](https://www.kaggle.com/suso172/arabic-natural-audio-dataset)

# Reading Data
- English Audio 
-- By using OS to read file from divice and creating Dataframe based on the following information from filenmae.
-- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
--Vocal channel (01 = speech, 02 = song).
--Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = --surprised).
--Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
--Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
--Repetition (01 = 1st repetition, 02 = 2nd repetition).
--Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
-- Their are 1440 Audio file used only 1200 file for train and test and execlude the rest for test.
- Arabic Audio
-- Audio were avilable in Kaggle as will as the dataframe with the extracted feature. 
-- I used the dataframe as the audio were not clear to do extracting feature with librosa.
-- Dataframe shape (1383, 847)

### Feature Extraction 
After inspicting random English audio speech file we can see all of them have silent of 0.5 second in the start so using librosa for feature extractig Audio using MFCC feature and trim silence in the beginning  and set rate sampling for the audio to 16000.

```py
for i in tqdm(range(len(data2_df))):
    X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',sr=16000,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[i] = [feature]
```

# Emotion Spectrogram 
- Diffrent male speech emotion
- Diffrent emotion between gender


# Key Finding (English Audio)
### Modeling 
- At first Develop the CNN model with Keras and constructed with 6 layers —  5 Conv1D layers followed by a Dense layer.
- Second, Constructed with 3 Conv1D layers, 2 MaxPooling1D layer, 2 BarchNormalization, 1 dropout layer and followed by a Dense layer.
- Then, Constructed with 4 Conv1D layers, 2 MaxPooling1D layer, 2 BarchNormalization , 2 dropout and 2 Dense layers.
- Lastly constructed with 3 Conv1D layers with regularizers, 2 BarchNormalization, 4 MaxPooling1D, 4 dropout layers, and 3 Dense layers.
### Data Augmentation: 
- [Adding White Noise](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html)
- Pitch Tuning.
- Speed and Pitch Tuning.

## Result:
The best performance when applied both L2 regularization and dropout regularization.
Data Augmentation (Adding White Noise & Pitch Tuning.)
-- Base Score 7%
-- Train Accuracy 97%
-- Test Accuracy 83%
```py
model = Sequential()

model.add(Conv1D(256, 8,padding='same', kernel_regularizer=regularizers.l2(0.01), activation='relu',
                 input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))

model.add(BatchNormalization())
model.add(Conv1D(128, 8, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.20))

model.add(Conv1D(64, 8, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Conv1D(64, 8, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())       
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.30))

model.add(Flatten())

model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.30))


model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.20))


model.add(Dense(y_train.shape[1], activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
```

##	Second Approach:
Simplify the model by reducing the target class:
- First reduce target to 5 emotion for both gender.
-- It show alittle improvment in the accuracy score.
-- Train accuracy 99%   Test accuracy 92%
- Second, reduce the target to Emotion only 
-- Train Accuracy 98%  & Test Accuracy 87%
- Third reduce target to only Gender without Emotion 
-- Train Accuracy 98%  Test Accuracy 94%

# Key Finding (Arabic Audo Data)
Developed the CNN model with Keras and constructed with 3 Conv1D layers with regularizers, 2 BarchNormalization, 4 MaxPooling1D, 4 dropout layers, and 3 Dense layers.
- Train Accuracy 99%
- Test Accuracy 94%
- Base Score 53%



# Project Summary
Inconclusion, The experiment went wall with classification of emotion by developing with CNN with Keras and aapplied both L2 regularization and dropout regularization. Moreover, Data Augmentation improve the score. Also, we can conclude that same emotion with different gender has different features and may not be easy to distangush between them with the current model as when inspected of some emotion we can see the simileraty in Man but differet in women.

# Live Demo
for live demo please refare to En_16Class notebook

# Futer Work 
Collecting more audio data and experiment with different model like RNN, experiment with different augmentation method and develop the model to predict live conversation.Making this project a tool for company and organizations to use like e-governments and Call center..etc. As monitoring and improving customer experience will help business retain revenue and earn new customer.

# About me:
GitHub: https://github.com/Mbuhliagah
Linkedin: https://www.linkedin.com/in/mbuhliagah-data-scientist/


