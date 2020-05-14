# function to clean a string
def clean(sentence):
    #remplace punctuation by blanck
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = {c: " " for c in filters}
    translate_map = str.maketrans(translate_dict)
    return sentence.translate(translate_map)

def load_data(train_file, test_file):
	train_texts, train_labels = [], []
	with open(train_file, "r") as f:
    	for line in f:
        	line = line.strip()
	        m = re.match("<\d+:\d+:(\w)> (.+)", line)
	        train_labels.append(1 if m.group(1)=="M" else -1 )
	        train_texts.append(m.group(2))
	test_texts = []
	with open(test_file, "r") as f:
	    for line in f:
	        line = line.strip()
	        m = re.match("<.+> (.+)", line)
	        test_texts.append(m.group(1))
	return train_texts, test_texts