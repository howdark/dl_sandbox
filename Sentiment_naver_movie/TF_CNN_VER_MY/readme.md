## Sentiment Classification with CNN using Tensorflow

#### Files

-   `data_handle_my.py` : collection of custom functions (data load, tokenize, labelling, padding and etc.)
-   `train_my.py` : script for CNN training
    -   parameters
    -   # Data loading params
    -   tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

    -   # Model Hyperparameters
    -   tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    -   tf.flags.DEFINE_integer("filter_size1", 5, "Filter sizes for conv-layer1")
    -   tf.flags.DEFINE_integer("filter_size2", 4, "Filter sizes for conv-layer2")
    -   tf.flags.DEFINE_integer("filter_size3", 3, "Filter sizes for conv-layer3")
    -   tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
    -   tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    -   tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    -   # Training parameters
    -   tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")   # success : 64
    -   # tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    -   tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")   # success : 200
    -   # tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    -   tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    -   tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    -   tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    -   # Misc Parameters
    -   tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    -   tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
-   `eval_my.py` : script for evaluation
    -   parameters
    -   # Eval Parameters
    -   tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
    -   # tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    -   tf.flags.DEFINE_string("checkpoint_dir", "./runs/1491221902/checkpoints", "Checkpoint directory from training run")  #149182239
    -   # tf.flags.DEFINE_string("checkpoint_dir", "./runs/1491097515/checkpoints", "Checkpoint directory from training run")
    -   tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
    -   # tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

    -   # Misc Parameters
    -   tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    -   tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#### Usage

###### Training

-   `$> python train_my.py`

###### Test (Evaluation)

-   `$> python eval_my.py`

#### Output (Only console print)

-   Training log (using `train.pickle`)
-   Test loss (using `test.pickle`)
-   Test accuracy (using `test.pickle`)
