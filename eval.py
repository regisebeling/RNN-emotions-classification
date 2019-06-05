import tensorflow as tf
import numpy as np
import os
import data_helpers
from sklearn import metrics
import csv

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("anger_dir", "data/rt-polaritydata/rt-polarity.angTest", "Path of anger data")
tf.flags.DEFINE_string("disgust_dir", "data/rt-polaritydata/rt-polarity.disgTest", "Path of disgust data")
tf.flags.DEFINE_string("fear_dir", "data/rt-polaritydata/rt-polarity.fearTest", "Path of fear data")
tf.flags.DEFINE_string("neutral_dir", "data/rt-polaritydata/rt-polarity.neutTest", "Path of neutral data")
tf.flags.DEFINE_string("sadness_dir", "data/rt-polaritydata/rt-polarity.sadTest", "Path of sadness data")
tf.flags.DEFINE_string("surprise_dir", "data/rt-polaritydata/rt-polarity.surpTest", "Path of surprise data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{} = {}".format(attr.upper(), value))
#print("")


def eval():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.anger_dir, FLAGS.disgust_dir, FLAGS.fear_dir, FLAGS.neutral_dir, FLAGS.sadness_dir, FLAGS.surprise_dir)

    # Substitution of code labels
    y_formatted = [];
    for label in y:
        if(label[0] == 1):
            y_formatted = np.concatenate([y_formatted, [0]]);
        elif(label[1] == 1):
            y_formatted = np.concatenate([y_formatted, [1]]);
        elif(label[2] == 1):
            y_formatted = np.concatenate([y_formatted, [2]]);
        elif(label[3] == 1):
            y_formatted = np.concatenate([y_formatted, [3]]);
        elif(label[4] == 1):
            y_formatted = np.concatenate([y_formatted, [4]]);
        elif(label[5] == 1):
            y_formatted = np.concatenate([y_formatted, [5]]);

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))
            print(metrics.classification_report(y_eval, all_predictions))
            # Save the evaluation to a csv
            predictions_human_readable = np.column_stack((np.array(x_text), all_predictions, y_formatted))
            out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
            	csv.writer(f).writerows(predictions_human_readable)

def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
