## Sentiment Classification with LSTM using KERAS

#### Files

-   `data_handle_keras_my.py` : collection of custom functions (data load, tokenize, labelling, padding and etc.)
-   `rnn_text_keras.py` : 2 layers of LSTM
    -   parameters
        -   dev_sample_percentage = 0.1 : "Percentage of the training data to use for validation"
        -   embedding_dim = 64
        -   learning_rate = 0.001
        -   hidden_unit = 64
        -   dropout_keep_prob = 0.5
        -   l2_reg_lambda = 0.0
        -   batch_size = 128
        -   num_epochs = 10

#### Usage

-   `$> python rnn_text_keras.py`

#### Output (Only console print)

-   Training log (using `train.pickle`)
-   Test loss (using `test.pickle`)
-   Test accuracy (using `test.pickle`)
