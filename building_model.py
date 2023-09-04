''' Insert this before the 'Building the recurrent neural network' comment in the second file and delete everything below.
After the model has been trained, you can return the code under the comment.
'''
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters)))) # 128 neurons
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01)) # lr = learning rate
model.fit(x, y, batch_size=256, epochs=4) # batch size = number of input-output pairs

model.save('textgen.model')
