import keras

# Callbacks are passed to the model via the callbacks
# argument in fit, which takes a list of callbacks.
# You can pass any number of callbacks.
callbacks_list = [
    # Interrupts training when improvement stops
    keras.callbacks.EarlyStopping(
        # Monitors the model's validation accuracy
        monitor = 'acc',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (two epochs)
        patience = 1,
    ),
    # Save the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath = 'my_model.h5',
        # These two arguments mean you won't overwrite
        # the model file unless val_loss has improved, which
        # allows you to keep the best model seen during training.
        monitor = 'val_loss',
        save_best_only = True,
    )
]

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              # You monitor accuracy, so it should be part
              # of the model's metrics
              metrics = ['acc'])

model.fit(x, y,
          epochs = 10,
          batch_size = 32,
          callbacks = callbacks_list,
          # The callback will monitor validation loss and
          # validation accuracy, so you need to pass 
          # validation_data to the call to fit.
          validation_data = (x_val, y_val))
