from collections import defaultdict

class trellis():
	def __init__ (self):
		self.word = ""
		self.tag = ""
		self.previous_tag = ""
		self.word_emission = 0.0
		self.probability = -1.0
		self.previous = None



def calculate_transition(tag_tag_count, tag_count):

	transmision_probability = {}
	for tag_tag in tag_tag_count:
		tag1_tag2_frequency = float(tag_tag_count[tag_tag])
		tag2_frequency = float(tag_count[tag_tag[1]])
		transmision_probability[tag_tag] = tag1_tag2_frequency / tag2_frequency

	return transmision_probability

def calculate_emission(word_tag_count, tag_count):

	emission_probability = {}
	for word_tag in word_tag_count:
		word_tag_frequency = float(word_tag_count[word_tag])
		tag_frequency = float(tag_count[word_tag[1]])
		emission_probability[word_tag] = word_tag_frequency / tag_frequency

	return emission_probability

def calculate_pobabilities():

	set_of_tags = set()
	set_of_tags.add("<s>") # <s> is begining of a sentence
	set_of_tags.add("</s>")

	word_tag_count = defaultdict(int)
	word_count = defaultdict(int)
	tag_count = defaultdict(int)

	tag_sequence = ["<s>"] 	# sequence of tags in a sentence
	tag_sequence_count = 0	# number of sentence of tag sequence
	tag_tag_count = defaultdict(int)

	for line in open("wsj00-18.tag.txt", 'r').readlines():
		line = line.strip()
		if "\t" in line:
			word_tag = line.split("\t")
			set_of_tags.add(word_tag[1])
			word_tag_count[(word_tag[0].lower(), word_tag[1])] += 1

			word_count[word_tag[0]] += 1
			tag_count[word_tag[1]] += 1

			tag_sequence.append(word_tag[1])
			if word_tag[1] == ".":
				tag_sequence.append("</s>")
				for i in range(0, len(tag_sequence)-1):
					tag_tag_count[(tag_sequence[i], tag_sequence[i+1])]	+= 1
				tag_sequence = ["<s>"]
				tag_sequence_count += 1

	tag_count["<s>"]  = tag_sequence_count
	tag_count["</s>"] = tag_sequence_count

	transition_probabilities = calculate_transition(tag_tag_count, tag_count)
	emission_probabilities = calculate_emission(word_tag_count, tag_count)
	set_of_tags = list(set_of_tags)

	return set_of_tags, emission_probabilities, transition_probabilities

def find_max_probability_for_first_word(transition, trellis):
	Transition = transition.get(("<s>", trellis.tag), 0)
	trellis.probability = Transition * trellis.word_emission
	trellis.previous_tag =  "<s>"



def find_max_probability(previous_trellis_list, transition, trellis):
	for item in previous_trellis_list:
		Transition = transition.get((item.tag, trellis.tag), 0)
		Probability = Transition * item.probability * trellis.word_emission
		if trellis.probability < Probability:
			trellis.probability = Probability
			trellis.previous_tag = item.tag
			trellis.previous = item


def viterbi(set_of_tags, emission, transition, sentence):
	matrix = []

	for i in range(0, len(sentence)):
		word = sentence[i].lower()

		if i == 0:
			temp = []
			for word_tag in emission:
				if word_tag[0] == word:
					empty_trellis = trellis()
					empty_trellis.word = word_tag[0]
					empty_trellis.tag = word_tag[1]
					empty_trellis.word_emission = emission.get(word_tag, 0)
					find_max_probability_for_first_word(transition, empty_trellis)
					temp.append(empty_trellis)
			matrix.append(temp)


		else:
			temp = []
			for word_tag in emission:
				if word_tag[0] == word:
					empty_trellis = trellis()
					empty_trellis.word = word_tag[0]
					empty_trellis.tag = word_tag[1]
					empty_trellis.word_emission = emission.get(word_tag, 0)
					find_max_probability(matrix[i-1], transition, empty_trellis)
					temp.append(empty_trellis)
			matrix.append(temp)



	list_of_tag = []
	last_trellis = matrix[len(matrix)-1][0]
	while last_trellis != None:
		list_of_tag.insert(0, last_trellis.tag)
		last_trellis = last_trellis.previous
	print(list_of_tag)



def main():
	set_of_tags, emission, transition = calculate_pobabilities()

	s1 = ['This','is','a','sentence','.']
	s2 = ['This','might','produce','a','result','if','the','system','works','well','.']
	s3 = ['Can','a','can','can','a','can','?']
	s4 = ['Can','a','can','move','a','can','?']
	s5 = ['Can','you','walk','the','walk','and','talk','the','talk','?']


	viterbi(set_of_tags, emission, transition, s1)
	viterbi(set_of_tags, emission, transition, s2)
	viterbi(set_of_tags, emission, transition, s3)
	viterbi(set_of_tags, emission, transition, s4)
	viterbi(set_of_tags, emission, transition, s5)

main()
