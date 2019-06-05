import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(anger_data_file, disgust_data_file, fear_data_file, neutral_data_file, sadness_data_file, surprise_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    anger_examples = list(open(anger_data_file, "r", encoding="UTF8").readlines())
    anger_examples = [s.strip() for s in anger_examples]
    sadness_examples = list(open(sadness_data_file, "r", encoding="UTF8").readlines())
    sadness_examples = [s.strip() for s in sadness_examples]
    surprise_examples = list(open(surprise_data_file, "r", encoding="UTF8").readlines())
    surprise_examples = [s.strip() for s in surprise_examples]
    fear_examples = list(open(fear_data_file, "r", encoding="UTF8").readlines())
    fear_examples = [s.strip() for s in fear_examples]
    disgust_examples = list(open(disgust_data_file, "r", encoding="UTF8").readlines())
    disgust_examples = [s.strip() for s in disgust_examples]
    neutral_examples = list(open(neutral_data_file, "r", encoding="UTF8").readlines())
    neutral_examples = [s.strip() for s in neutral_examples]

    # Split by words
    x_text = anger_examples + disgust_examples + fear_examples + neutral_examples + sadness_examples + surprise_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    anger_labels = [[0, 0, 0, 0, 0, 1] for _ in anger_examples]
    disgust_labels = [[0, 0, 0, 0, 1, 0] for _ in disgust_examples]
    fear_labels = [[0, 0, 0, 1, 0, 0] for _ in fear_examples]
    neutral_labels = [[0, 0, 1, 0, 0, 0] for _ in neutral_examples]
    sadness_labels = [[0, 1, 0, 0, 0, 0] for _ in sadness_examples]
    surprise_labels = [[1, 0, 0, 0, 0, 0] for _ in surprise_examples]
    y = np.concatenate([anger_labels, disgust_labels, fear_labels, neutral_labels, sadness_labels, surprise_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    pos_dir = "data/rt-polaritydata/rt-polarity.pos"
    neg_dir = "data/rt-polaritydata/rt-polarity.neg"

    load_data_and_labels(pos_dir, neg_dir)
