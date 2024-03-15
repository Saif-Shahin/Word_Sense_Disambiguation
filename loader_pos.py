'''

Adapted from the work of jcheung

'''
import xml.etree.ElementTree as ET # was xml.etree.cElementTree
import codecs

class WSDInstance:
    def __init__(self, my_id, lemma, context, index, pos):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
        self.pos = pos          # POS of the ambiguous word
    def __str__(self):
        return '%s\t%s\t%s\t%d\t%s' % (self.id, self.lemma, ' '.join(self.context), self.index, self.pos)

def load_instances(f):
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    pos = el.attrib['pos'] # Extract POS information
                    instances[my_id] = WSDInstance(my_id, lemma, context, i, pos)
    return dev_instances, test_instances


def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    '''
    dev_key = {}
    test_key = {}
    with open(f) as file:
     for line in file:
        if len(line) <= 1: continue
        #print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')
