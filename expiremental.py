import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import torch
#from datasets import load_dataset
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel
from sklearn.preprocessing import LabelEncoder
import numpy as np


#import hydra.utils as hu

#from omegaconf import OmegaConf

#from fairseq.models.roberta import RobertaModel


import loader
import bootstrap
from transformers import RobertaTokenizer
from nltk.corpus import wordnet as wn

def count_wordnet_senses(term):
    synsets= wn.synsets(term)
    num_senses = len(synsets)
    return num_senses


def plotssandpp(accuracyss, accuracypp):
        # Data
    methods = ['Lemmatization + Stopwords', 'Stopwords Only']
    accuracies = [accuracypp, accuracyss]  # Example accuracies

        # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['blue', 'green'])
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('RoBERTa Accuracies for "Rule"')
    plt.ylim(0, 1.1)  # Y-axis from 0 to 1
    plt.yticks([i * 0.05 for i in range(21)])
    plt.show()



def main():
    dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words = bootstrap.setup_and_load_data()
    wsd_terms= bootstrap.get_most_frequent_lemmas(dev_instances, test_instances)

    wsd_terms.append("support")
    wsd_terms.append("law")
    wsd_terms.append("rule")
    wsd_terms.append("end")

    train_data_folder= 'SemCor'

    #note: SemCor sentences are mapped from wordnet 3.0, hence I will determine the number of senses per word based on that
    models_ls={}
    pp= "Lemmatization + Stop Words"

    wsd_terms_fast=[]
    wsd_terms_fast.append(wsd_terms[0])
    wsd_terms_fast  .append(wsd_terms[1])
    wsd_terms_fast.append("law")
    wsd_terms_fast.append("rule")

    training_datapp = load_and_filter_training_data("rule", train_data_folder, use_lemmatization=True,
                                                  remove_stop_words=True)

    unique_labels = set()
    for instance in training_datapp: #WHAT IS AN INSTANCE IN TRAIN INSTANCES?
        unique_labels.add(instance.value)

    unique_labels_list = list(unique_labels)

    modelpp, tokenizerpp, label_encoderpp = train_experemental_model(term="rule", dev_instances=dev_instances, dev_key=dev_key,
                                                               pp=pp, num_senses=count_wordnet_senses("rule"),
                                                               train_instances=training_datapp, lemmatizer=lemmatizer,
                                                               stop_words=stop_words, epochs=3)


    accuracypp = evaluate_model_accuracy(model=modelpp, term="rule", instances=test_instances, keys=test_key,
                                       tokenizer=tokenizerpp, label_encoder=label_encoderpp, lemmatizer=lemmatizer,
                                       stop_words=stop_words, unique_labels_list=unique_labels_list)

    training_datass = load_and_filter_training_data("rule", train_data_folder,
                                                    remove_stop_words=True)

    unique_labels = set()
    for instance in training_datapp:  # WHAT IS AN INSTANCE IN TRAIN INSTANCES?
        unique_labels.add(instance.value)

    unique_labels_list = list(unique_labels)

    modelss, tokenizerss, label_encoderss = train_experemental_model(term="rule", dev_instances=dev_instances,
                                                                     dev_key=dev_key,
                                                                     pp=pp, num_senses=count_wordnet_senses("rule"),
                                                                     train_instances=training_datass,
                                                                     stop_words=stop_words, epochs=3)

    accuracyss = evaluate_model_accuracy(model=modelss, term="rule", instances=test_instances, keys=test_key,
                                         tokenizer=tokenizerss, label_encoder=label_encoderss,
                                         stop_words=stop_words, unique_labels_list=unique_labels_list)

    plotssandpp(accuracyss=accuracyss, accuracypp=accuracypp)

    for term in wsd_terms_fast:
        training_data= load_and_filter_training_data(term, train_data_folder,  use_lemmatization=True, remove_stop_words=True )
        model, tokenizer, label_encoder = train_experemental_model(term=term, dev_instances= dev_instances, dev_key=dev_key , pp=pp,  num_senses=count_wordnet_senses(term), train_instances=training_data, lemmatizer=lemmatizer, stop_words=stop_words,epochs=3)
        models_ls[term] = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder
        }


    pp= "Stop Words Only"
    models_sw={}
    for term in wsd_terms_fast:
        training_data= load_and_filter_training_data(term, train_data_folder, remove_stop_words=True )
        model, tokenizer, label_encoder= train_experemental_model(term=term,dev_instances= dev_instances, dev_key=dev_key , pp=pp, num_senses=count_wordnet_senses(term), train_instances=training_data, stop_words=stop_words, epochs=3)
        models_sw[term] = {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder
        }

    training_data = load_and_filter_training_data("rule", train_data_folder, use_lemmatization=True, remove_stop_words=True)
    model_rule_lasso_LS, tokenizer_rule_lasso_LS, label_encoder_rule_lasso_LS= train_experemental_model(term="rule", dev_instances= dev_instances, dev_key=dev_key, num_senses=count_wordnet_senses("rule"), train_instances=training_data, stop_words=stop_words, epochs=3,lasso=True, pp="Lasso, Lemmatization + Stopwords")

    unique_labels = set()
    for instance in training_data: #WHAT IS AN INSTANCE IN TRAIN INSTANCES?
        unique_labels.add(instance.value)

    unique_labels_list = list(unique_labels)

    accuracy_rule_lasso_ls = evaluate_model_accuracy(model= model_rule_lasso_LS, instances= test_instances, keys=test_key, tokenizer= tokenizer_rule_lasso_LS, label_encoder=label_encoder_rule_lasso_LS, term="rule",lemmatizer=lemmatizer, stop_words=stop_words, unique_labels_list=unique_labels_list )




    # Evaluate models in models_ls
    accuracies_ls = {}
    for term, components in models_ls.items():
        model = components['model']
        tokenizer = components['tokenizer']
        label_encoder = components['label_encoder']
        unique_labels = set()
        for instance in training_data:  # WHAT IS AN INSTANCE IN TRAIN INSTANCES?
            unique_labels.add(instance.value)

        unique_labels_list = list(unique_labels)
        accuracy = evaluate_model_accuracy(model= model,term=term, instances= test_instances, keys= test_key, tokenizer= tokenizer, label_encoder=label_encoder, lemmatizer=lemmatizer, stop_words=stop_words, unique_labels_list= unique_labels_list)
        accuracies_ls[term] = accuracy

    # Evaluate models in models_sw
    accuracies_sw = {}
    for term, components in models_sw.items():
        model = components['model']
        tokenizer = components['tokenizer']
        label_encoder = components['label_encoder']
        for instance in training_data:  # WHAT IS AN INSTANCE IN TRAIN INSTANCES?
            unique_labels.add(instance.value)

        unique_labels_list = list(unique_labels)
        accuracy = evaluate_model_accuracy(model= model,term=term, instances= test_instances, keys= test_key, tokenizer= tokenizer, label_encoder=label_encoder, stop_words=stop_words, unique_labels_list= unique_labels_list)
        accuracies_sw[term] = accuracy

    accuracy_rule_lasso_sw=0

    plot_accuracies(accuracies_ls, accuracies_sw, accuracy_rule_lasso_ls, accuracy_rule_lasso_sw)


def load_and_filter_training_data(wsd_term, train_data_folder, include_brown=True, max_per_wsd_term=250, use_lemmatization=False, remove_stop_words=False):
    training_instances = []
    term_count = 0
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None
    stop_words = set(stopwords.words('english')) if remove_stop_words else None
    wsd_term_words = wsd_term.lower().split()

    # Function to parse labels from .txt file
    def parse_labels(label_file):
        label_map = {}
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    identifier, label = parts
                    label_map[identifier] = label
        return label_map

    # Function to process a sentence and add TrainInstances
    # Function to process a sentence and add TrainInstances
    def process_sentence(words, labels):
        nonlocal term_count
        for index, word in enumerate(words):
            word_lower = word.lower()
            word_lower=word_lower.replace("--", " ")

            processed_word = lemmatizer.lemmatize(word_lower) if lemmatizer else word_lower
            if remove_stop_words and processed_word in stop_words:
                continue

            # Check for single-word match or start of multi-word match
            if processed_word == wsd_term_words[0]:
                if len(wsd_term_words) == 1 or (len(words) - index >= len(wsd_term_words) and
                                                all(lemmatizer.lemmatize(words[index + i].lower()) if lemmatizer else
                                                    words[index + i].lower() == (lemmatizer.lemmatize(wsd_term_words[i]) if lemmatizer else wsd_term_words[i]) for i in
                                                    range(len(wsd_term_words)))):
                    identifier = word_ids[index] #maby this is the bug
                    label = labels.get(identifier, None)
                    if label:
                        context = [w.lower() for w in words]  # Convert all context words to lowercase
                        training_instances.append(TrainInstance(processed_word, context, index, label))
                        term_count += 1
                        if term_count >= max_per_wsd_term:
                            return True
        return False

    if include_brown:
        for filename in os.listdir(train_data_folder):
            if filename.endswith(".xml"):
                xml_path = os.path.join(train_data_folder, filename)
                txt_path = "SemCor/semcor.gold.key.txt"  # Assuming this is the correct path
                labels = parse_labels(txt_path)

                tree = ET.parse(xml_path)
                root = tree.getroot()

                for sentence in root.iter('sentence'):
                    sentence_words = []
                    word_ids = []  # To keep track of word IDs
                    for elem in sentence:
                        if elem.tag == 'wf' or elem.tag == 'instance':
                            word_text = elem.text.lower()  # Convert word to lowercase
                            identifier = elem.get('id', '')  # Extract the identifier
                            if not stop_words or word_text not in stop_words:
                                word_text = lemmatizer.lemmatize(word_text) if lemmatizer else word_text
                                sentence_words.append(word_text)
                                word_ids.append(identifier)  # Add the identifier to word_ids
                    if process_sentence(sentence_words, labels):
                        return training_instances

    return training_instances



#function to convert my training instances into a format suitable for roberta
class WSDataset(Dataset):
    def __init__(self, instances, tokenizer, label_encoder, max_length=512):
        self.instances = instances
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        # Return the total number of instances in the dataset
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        # Tokenize the sentence
        encoded = self.tokenizer.encode_plus(
            instance.context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # Encode the label
        label = self.label_encoder.transform([instance.value])[0]

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label)

class WSDClassifier(nn.Module):
    def __init__ (self, roberta, num_senses):
        super(WSDClassifier, self).__init__()
        self.roberta =roberta
        self.shared_layer = nn.Linear(roberta.config.hidden_size, roberta.config.hidden_size)
        self.attention= nn.Linear(roberta.config.hidden_size,1)
        self.classifier = nn.Linear(roberta.config.hidden_size, num_senses)

    def forward(self, input_ids, attention_mask):

        #pass the input through roberta to get the hidden state of last layer
        outputs = self.roberta(input_ids=input_ids, attention_mask= attention_mask)
        seq_output= outputs.last_hidden_state

        #apply attention mechnisim to seq output to learn which parts of the seq are important for WSD
        attention_weights = torch.softmax(self.attention(seq_output), dim=1)
        weighted_seq_out= seq_output *attention_weights

        #aggregating the seq representation to turn it into one vector that represents the context
        aggregated_representation= torch.sum(weighted_seq_out, dim=1)

        #seq representation is passed through classifier to predict the sense
        logits = self.classifier(aggregated_representation)

        return logits

class TrainInstance:
    def __init__(self, lemma, context, index, value):
        self.lemma = lemma      # Lemma of the word whose sense is to be resolved
        self.context = context  # List of all the words in the sentential context #todo: bug! contex has the identifier, but shouldnt.
        self.index = index      # Index of lemma within the context
        self.value= value


def normalize_label(label):
    return label.replace("_", " ").lower()

def train_experemental_model(term, train_instances,num_senses,  dev_key, dev_instances, pp,stop_words=None, lasso=False, lemmatizer=None ,full=False, epochs= 5):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Initialize RoBERTa and WSDClassifier
    label_encoder = LabelEncoder()
    roberta = RobertaModel.from_pretrained('roberta-base')

    unique_labels = set()
    for instance in train_instances: #WHAT IS AN INSTANCE IN TRAIN INSTANCES?
        unique_labels.add(instance.value)

    unique_labels_list = list(unique_labels)

    label_encoder.fit(unique_labels_list) #encoded the labels of the WSD term that are the train data

    model = WSDClassifier(roberta, len(unique_labels_list))

    # Prepare the dataset and dataloader
    dataset = WSDataset(train_instances, tokenizer, label_encoder)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    lasso_lambda = 0.01

    dev_accuracies = []

    for epoch in range(epochs):
        model.train()
        for input_ids, attention_masks, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
#todo: test functionality when wsd term is something like united states
            if lasso:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += lasso_lambda * l1_penalty

            loss.backward()
            optimizer.step()

        dev_accuracy = evaluate_model_accuracy(model, unique_labels_list=unique_labels_list, instances=dev_instances, keys= dev_key, tokenizer= tokenizer, label_encoder= label_encoder, term=term, lemmatizer=lemmatizer,  stop_words=stop_words,
                                               full=False)

        dev_accuracies.append(dev_accuracy)

        print(f"Term: {term}. Epoch {epoch + 1}, Training Loss: {loss.item()}, Development Accuracy: {dev_accuracy}")

    if dev_accuracies:
        # Plot if dev_accuracies is not empty
        plt.plot(range(1, epochs + 1), dev_accuracies, marker='o', label='Development Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Development Accuracy Over Epochs for {pp}')
        plt.legend()
        plt.xticks(range(1, epochs + 1))
        plt.show()
    else:
        # Handle the case where dev_accuracies is empty
        print("No data available to plot for dev_accuracies because the wsd term is not in the dev list.")

    return model, tokenizer, label_encoder


def evaluate_model_accuracy(model, unique_labels_list,instances, keys, tokenizer, label_encoder, term, lemmatizerC=False, lemmatizer=None,  stop_words= None, full=False):
    model.eval()
    correct_predictions, total_predictions = 0, 0
    accuracy=0

    # Normalize the term for comparison (handling multi-word phrases)
    def normalize_term(t):
        return t.replace("_", " ").lower()

    normalized_term = normalize_label(term)

    with torch.no_grad():
        for instance_id, wsd_instance in instances.items():
            # Normalize the lemma for comparison
            normalized_lemma = normalize_label(wsd_instance.lemma)

            # Filter instances based on the term when full is False
            if not full and normalized_lemma != normalized_term:
                continue

            # Proceed with evaluation
            input_ids, attention_mask = process_wsd_instance(wsd_instance, tokenizer, label_encoder,lemmatizer, stop_words ) #the lemmas are not being preprocessed!(case wise)
            outputs = model(input_ids, attention_mask) #outputs=tensor([[0.1310]])
            probabilities = torch.softmax(outputs, dim=1) #probabiliites:tensor([[1.]])
            predicted_label_idx = torch.argmax(probabilities, dim=1).item() #predicted_label_idx =0



            # Get true labels (all possible correct keys), ensuring case-insensitivity
            true_labels = [label for label in keys.get(instance_id, [])]
            for label in true_labels:
                if label not in unique_labels_list:
                    print(f"The sense '{label}' not present in train data!")

                else:
                    true_label_idxs = label_encoder.transform(true_labels)
                    if predicted_label_idx in true_label_idxs:
                        correct_predictions += 1

            total_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    if ((term == "Rule") and (lemmatizerC== True)):
        print("Accuracy of 'Rule' (for LASSO evaluation): ", accuracy)
    return accuracy

def process_wsd_instance(wsd_instance, tokenizer, label_encoder, lemmatizer=None, stop_words=None):
    # Normalize and handle multi-word phrases in lemma
    wsd_instance.lemma = wsd_instance.lemma.replace("_", " ").lower()

    # Process context words
    processed_context = []
    for word in wsd_instance.context:
        word_lower = word.lower()
        word_lower= word_lower.replace("--", " ")

        # Lemmatize if lemmatizer is provided
        if lemmatizer:
            word_lower = lemmatizer.lemmatize(word_lower)

        # Add word to processed_context if not a stop word or if stop word removal is not applied
        if not stop_words or word_lower not in stop_words:
            processed_context.append(word_lower)

    # Tokenize the sentence context
    encoded = tokenizer.encode_plus(
        ' '.join(processed_context),
        add_special_tokens=True,
        max_length=512,  # Adjust as needed
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0)

    # Encode the label


    return input_ids, attention_mask

#todo: finish the development testing and the total model evaluation on two words and one epoch (small example)
#Then, finish the graph functions and see if it works correclty.
#then, run it for all words and leave it running overnight

#todo: perform lemmatization and stemming (when approprpriate) on dev and test data.
#todo: see possible issue of vlasses being keys, not lemmas in label encoder
#todo: make code work for multi word lemmas like united states

def plot_accuracies(accuracies_ls, accuracies_sw, accuracy_rule_lasso_ls, accuracy_rule_lasso_sw):
    # Names of terms plus the special 'Rule' term for Lasso
    terms = list(accuracies_ls.keys()) + ['Rule_Lasso']

    # Accuracies for lemmatization and stopwords (including special case)
    ls_accuracies = [accuracies_ls.get(term, accuracy_rule_lasso_ls if term == 'Rule + LASSO Reg.' else 0) for term in terms]

    # Accuracies for stopwords only (including special case)
    sw_accuracies = [accuracies_sw.get(term, accuracy_rule_lasso_sw if term == 'Rule + LASSO Reg.' else 0) for term in terms]


    # Calculate averages
    avg_ls = sum(ls_accuracies) / len(ls_accuracies)
    avg_sw = sum(sw_accuracies) / len(sw_accuracies)

    # Setting the positions and width for the bars
    x = np.arange(len(terms))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, ls_accuracies, width, label='Lemmatization + Stopwords')
    rects2 = ax.bar(x + width/2, sw_accuracies, width, label='Stopwords Only')

    # Add average lines
    ax.axhline(y=avg_ls, color='blue', linestyle='--', label=f'Avg Lemmatization + Stopwords: {avg_ls:.2f}')
    ax.axhline(y=avg_sw, color='orange', linestyle='--', label=f'Avg Stopwords Only: {avg_sw:.2f}')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('RoBERTa: Accuracies by Term and Preprocessing Method')
    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.legend()

    # Add grid lines for better readability
    ax.yaxis.grid(True)

    # Attach a text label above each bar in rects1 and rects2, displaying its height.
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.show()







#TODO: this function SHOULD BE DELETED AND ENTRY SHOULD BE FIXED



if __name__ == '__main__':
    main()

