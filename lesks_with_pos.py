import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from loader_pos import load_instances, load_key, WSDInstance
import matplotlib.pyplot as plt
from nltk import pos_tag


def setup_and_load_data():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    data_file = 'multilingual-all-words.en.xml'
    key_file = 'wordnet.en.key'

    dev_instances, test_instances = load_instances(data_file)
    dev_key, test_key = load_key(key_file)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    print("The number of dev instances: ",  len(dev_instances))  # number of dev instances
    print("The number of test instances: ", len(test_instances))  # number of test instances

    return dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words


def preprocess_l_sw(context, lemmatizer, stop_words):
    tokens = word_tokenize(context)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return [word for word in lemmatized if word not in stop_words]


def preprocess_l_sw_with_pos(context, lemmatizer, stop_words):
    tokens = word_tokenize(context)
    pos_tokens = pos_tag(tokens)
    lemmatized = [(lemmatizer.lemmatize(token), tag) for token, tag in pos_tokens]
    return [(word, tag) for word, tag in lemmatized if word not in stop_words]

def preprocess_sw_with_pos(context, stop_words):
    tokens = word_tokenize(context)
    pos_tokens = pos_tag(tokens)
    return [(word, tag) for word, tag in pos_tokens if word not in stop_words]


def preprocess_sw(context, stop_words):
    tokens = word_tokenize(context)
    return [word for word in tokens if word not in stop_words]

def lesk_algorithm_scratch(context_sentence, ambiguous_word, lemmatizer, stop_words):
    best_sense = None
    max_overlap = 0
    context = set(preprocess_l_sw(context_sentence, lemmatizer, stop_words))

    for sense in wn.synsets(ambiguous_word):
        # Definition and examples for the sense
        signature = word_tokenize(sense.definition())
        for example in sense.examples():
            signature += word_tokenize(example)
        signature = set(preprocess_l_sw(' '.join(signature), lemmatizer, stop_words))

        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


def lesk_algorithim_NLTK(context_sentence, ambiguous_word, lemmatizer, stop_words):
    preprocessed_context = preprocess_l_sw(context_sentence, lemmatizer, stop_words)
    best_sense = lesk(preprocessed_context, ambiguous_word)
    return best_sense


def most_frequent_sense(lemma):
    synsets = wn.synsets(lemma)
    return synsets[0] if synsets else None

def lesk_algorithm_scratch_sw(context_sentence, ambiguous_word, stop_words):
    best_sense = None
    max_overlap = 0
    context = set(preprocess_sw(context_sentence, stop_words))

    for sense in wn.synsets(ambiguous_word):
        signature = word_tokenize(sense.definition())
        for example in sense.examples():
            signature += word_tokenize(example)
        signature = set(preprocess_sw(' '.join(signature), stop_words))

        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense

def lesk_algorithm_NLTK_sw(context_sentence, ambiguous_word, stop_words):
    preprocessed_context = preprocess_sw(context_sentence, stop_words)
    best_sense = lesk(preprocessed_context, ambiguous_word)
    return best_sense

def lesk_algorithm_with_pos(context_sentence, ambiguous_word, ambiguous_word_pos, lemmatizer, stop_words):
    best_sense = None
    max_overlap = 0
    context = set(preprocess_l_sw_with_pos(context_sentence, lemmatizer, stop_words))

    # Map the POS tag of the ambiguous word to WordNet format
    wn_pos = get_wordnet_pos(ambiguous_word_pos)

    for sense in wn.synsets(ambiguous_word):
        # Check if sense POS matches the ambiguous word POS
        if sense.pos() != wn_pos:
            continue

        # Definition and examples for the sense
        signature = word_tokenize(sense.definition())
        for example in sense.examples():
            signature += word_tokenize(example)
        signature = set(preprocess_l_sw_with_pos(' '.join(signature), lemmatizer, stop_words))

        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense

def get_wordnet_pos(treebank_tag):
    """
    Map 'treebank' POS tags to WordNet POS tags.
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def evaluate_wsd(instances, keys, lemmatizer, stop_words):
    correct_mfs = 0
    correct_lesk_nltk = 0
    correct_lesk_scratch = 0
    correct_lesk_nltk_sw = 0
    correct_lesk_scratch_sw = 0
    correct_lesk_pos = 0
    total = len(instances)

    for instance_id, instance in instances.items():
        context_sentence = ' '.join(instance.context)
        correct_sense_keys = keys[instance_id]

        ambiguous_word_pos = instance.pos

        # Lesk Algorithm with POS
        lesk_custom_sense_pos = lesk_algorithm_with_pos(context_sentence, instance.lemma, ambiguous_word_pos,
                                                        lemmatizer, stop_words)
        if lesk_custom_sense_pos and any(lemma.key() in correct_sense_keys for lemma in lesk_custom_sense_pos.lemmas()):
            correct_lesk_pos += 1

        # Most Frequent Sense
        mfs_sense = most_frequent_sense(instance.lemma)
        if mfs_sense and any(lemma.key() in correct_sense_keys for lemma in mfs_sense.lemmas()):
            correct_mfs += 1

        # NLTK's Lesk Algorithm
        lesk_nltk_sense = lesk_algorithim_NLTK(context_sentence, instance.lemma, lemmatizer, stop_words)
        if lesk_nltk_sense and any(lemma.key() in correct_sense_keys for lemma in lesk_nltk_sense.lemmas()):
            correct_lesk_nltk += 1

        # Custom Lesk Algorithm
        lesk_custom_sense = lesk_algorithm_scratch(context_sentence, instance.lemma, lemmatizer, stop_words)
        if lesk_custom_sense and any(lemma.key() in correct_sense_keys for lemma in lesk_custom_sense.lemmas()):
            correct_lesk_scratch += 1

        # NLTK's Lesk Algorithm without Lemmatization
        lesk_nltk_sense_sw = lesk_algorithm_NLTK_sw(context_sentence, instance.lemma, stop_words)
        if lesk_nltk_sense_sw and any(lemma.key() in correct_sense_keys for lemma in lesk_nltk_sense_sw.lemmas()):
            correct_lesk_nltk_sw += 1

        # Custom Lesk Algorithm without Lemmatization
        lesk_custom_sense_sw = lesk_algorithm_scratch_sw(context_sentence, instance.lemma, stop_words)
        if lesk_custom_sense_sw and any(lemma.key() in correct_sense_keys for lemma in lesk_custom_sense_sw.lemmas()):
            correct_lesk_scratch_sw += 1

    return {
        'mfs_accuracy': correct_mfs / total,
        'lesk_pos_accuracy': correct_lesk_pos / total,
        'lesk_nltk_accuracy': correct_lesk_nltk / total,
        'lesk_from_scratch_accuracy': correct_lesk_scratch / total,
        'lesk_nltk_accuracy_sw': correct_lesk_nltk_sw / total,
        'lesk_from_scratch_accuracy_sw': correct_lesk_scratch_sw / total
    }





def run_wsd_evaluation(dev_instances, test_instances,dev_key, test_key, lemmatizer, stop_words):
    dev_accuracies = evaluate_wsd(dev_instances, dev_key, lemmatizer, stop_words)
    print(f"Development Set - Most Frequent Sense Accuracy: {dev_accuracies['mfs_accuracy']}")
    print(f"Development Set - POS Lesk Accuracy: {dev_accuracies['lesk_pos_accuracy']}")
    print(f"Development Set - NLTK Lesk Accuracy: {dev_accuracies['lesk_nltk_accuracy']}")
    print(f"Development Set - Lesk From Scratch Accuracy: {dev_accuracies['lesk_from_scratch_accuracy']}")
    print(f"Development Set - NLTK Lesk No Lemmatization Accuracy: {dev_accuracies['lesk_nltk_accuracy_sw']}")
    print(f"Development Set - Lesk From Scratch No Lemmatization Accuracy: {dev_accuracies['lesk_from_scratch_accuracy_sw']}")

    # Evaluate on test set
    test_accuracies = evaluate_wsd(test_instances, test_key, lemmatizer, stop_words)
    print(f"Test Set - Most Frequent Sense Accuracy: {test_accuracies['mfs_accuracy']}")
    print(f"Test Set - POS Lesk Accuracy: {test_accuracies['lesk_pos_accuracy']}")
    print(f"Test Set - NLTK Lesk Accuracy: {test_accuracies['lesk_nltk_accuracy']}")
    print(f"Test Set - Lesk From Scratch Accuracy: {test_accuracies['lesk_from_scratch_accuracy']}")
    print(f"Test Set - NLTK Lesk No Lemmatization Accuracy: {test_accuracies['lesk_nltk_accuracy_sw']}")
    print(f"Test Set - Lesk From Scratch No Lemmatization Accuracy: {test_accuracies['lesk_from_scratch_accuracy_sw']}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(dev_accuracies.keys()), list(dev_accuracies.values()), marker='o', label='Development Set')
    plt.plot(list(test_accuracies.keys()), list(test_accuracies.values()), marker='o', label='Test Set')
    plt.xlabel('WSD Method')
    plt.ylabel('Accuracy')
    plt.title('Comparison of WSD Methods')
    plt.xticks(rotation=45)
    plt.yticks([i * 0.05 for i in range(21)])  # Y-axis increments of 0.05
    plt.grid(axis='y')  # Adding horizontal grid lines on the y-axis
    plt.legend()
    plt.tight_layout()
    plt.savefig("lesks_with_pos.png")
    plt.show()


def main():
    dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words = setup_and_load_data()
    run_wsd_evaluation(dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words)


if __name__ == '__main__':
    main()