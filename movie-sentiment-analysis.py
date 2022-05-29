# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Ayan Abhiranya Singh
# CS 156, Assignment 4C

import json  # to build an output file with results
import math  # to perform mathematical operations like taking log
import os  # to read the file names in directory


# def clean_text(text) removes specific punctuations from code to clean it up as part of pre-processing
# this function also converts all forms to lower-case, to equate them to words in the generated model.
def clean_text(text):
    delete_chars = {'"', '*', '!', '(', ')', '+', '.', '?', '/', '<', '>', '@', '^', '_', '`', '{', '|', '~', ','}
    cleaned_text = ""
    for char in text:  # for each character in the text stream, check if it is one of the punctuation marks
        if char not in delete_chars:
            cleaned_text += char.lower()  # otherwise it is a word, so convert it to lower case
    return cleaned_text.split()


# def preprocess returns the total count of words of a given class as well as the number of documents of that class
def preprocess(directory, class_type):
    word_appearances = {}

    words_in_file = {}

    #  ignore_words defines the words that need to be skipped as part of the pre-processing
    #  this includes stop words and some other common words that provide no information on review sentiment
    ignore_words = {'the', 'this', 'that', 'there', 'they', 'their', 'for', 'who',
                    'and', 'is', 'to', 'in', 'of', 'be', 'a', 'an', 'as', 'so', 'what',
                    'you', 'i', 'about', 'has', 'had', 'it', 'its', 'his', 'with', 'her', 'hers'}

    non_relevant_words = {'religion', 'avoids', 'offbeat', 'annual', 'winslet', 'jolie', 'addresses',
                          'animators', 'gump', 'chuckle', 'schumacher', 'seagal',
                          'hudson', 'angelina', 'furniture', 'page', 'scorsese',
                          'fashioned', 'studies', 'tarantino', 'elliot'}

    filecount = 0
    for filename in os.listdir(directory):  # for each filename in the specified directory
        filepath = os.path.join(directory, filename)  # generate filepath by concatenating directory and filename
        if filepath.endswith(".txt"):  # if it is a txt file (format of files provided), then open the file
            f = open(filepath, "r")  # open the file and start pre-processing
            words = clean_text(f.read())  # remove unnecessary punctuation from the file
            words_in_file = {}
            for word in words:  # for each word in the file
                if word in ignore_words:  # if the word is an ignore_word, then skip it
                    continue
                if word in non_relevant_words:  # if the word is a non_relevant_word, then skip it
                    continue
                if word not in words_in_file:  # if the word has already been added to our model, then skip it
                    words_in_file[word] = 1  # following the formula for mutual information,
                    # we only want the count of how many times a word appears in all of the review files,
                    # not the total count.

            # create a list that contains counts of all word appearances across all files of the directory
            # this list should be dumped into a file
            for word in words_in_file:  # if the word has been counted in another file,
                if word in word_appearances:
                    word_appearances[word] += 1  # increment it for the current file
                else:
                    word_appearances[word] = 1  # add the word to the file, if not.

            filecount = filecount + 1  # increment file_count by 1
            f.close()

    threshold_for_appearances = 0.50 * filecount

    word_appearances = dict(sorted(word_appearances.items(), key=lambda item: item[1], reverse=True))
    # create a file containing the word counts for the class type
    word_count_file = "word-count-" + class_type + ".txt"
    output = open(word_count_file, "w")
    for word in word_appearances:
        if word_appearances[word] > threshold_for_appearances:
            output.write(json.dumps(word) + ': ' + str(word_appearances[word]) + '\n')

    output.close()

    # return both the count of word appearances and the total file count
    return word_appearances, filecount


#  get_mutual_information calculates the mutual information for each attribute (word) of the Naive Bayes Classifier
def get_mutual_information(pos_dir, neg_dir):
    pos_file_count = 0
    neg_file_count = 0

    pos_word_appearances = {}  # to hold the return value from def pre_process
    neg_word_appearances = {}  # to hold the return value from def pre_process

    pos_word_information = {}  # to hold the information of each word in the class pos
    neg_word_information = {}  # to hold the information of each word in the class neg

    # call preprocess to get word counts for both positive and negative directories
    pos_word_appearances, pos_file_count = preprocess(pos_dir, "pos")
    neg_word_appearances, neg_file_count = preprocess(neg_dir, "neg")

    # add up the positive and negative file counts to get the total file count
    total_file_count = pos_file_count + neg_file_count

    # add up the positive and negative file counts to get the total file count
    for word in pos_word_appearances:  # calculate the mutual information for each word of the positive files
        total_appearances = pos_word_appearances[word]  # calculate total appearances of the word as postive
        if word in neg_word_appearances:  # plus negative appearances
            total_appearances += neg_word_appearances[word]
        # compute result as word appearance within the specified class * total number of files
        # divided by the total appearances of the word * count of files of the specified class
        result = (pos_word_appearances[word] * total_file_count) / (total_appearances * pos_file_count)
        pos_word_information[word] = math.log2(result)  # store the result as log base of 2

    # repeat the above process for negative word appearances
    for word in neg_word_appearances:
        total_appearances = neg_word_appearances[word]
        if word in pos_word_appearances:
            total_appearances += pos_word_appearances[word]
        result = (neg_word_appearances[word] * total_file_count) / (total_appearances * neg_file_count)
        neg_word_information[word] = math.log2(result)

    return pos_word_information, neg_word_information


# return most useful attributes in the information, thus building an optimized set of attributes for
# classification
def get_useful_info(first_info, second_info):
    useful_info = {}  # optimized version of the info being filtered for most important information
    usefulness = 0
    usefulness_threshold = 2  # threshold to select only those attributes that have a certain strength

    # usefulness is computed as the absolute value of
    # the difference between the probability of A | pos and A | neg
    for word in first_info:  # compute the usefulness for all words in the information dictionaries
        if word not in second_info:
            usefulness = first_info[word]
        else:
            usefulness = first_info[word] - second_info[word]
        if usefulness > usefulness_threshold:  # store this
            useful_info[word] = first_info[word]

    useful_info = dict(sorted(useful_info.items(), key=lambda item: item[1], reverse=True))

    return useful_info


# to generate a single classifier file for sentiment analysis
# combine both models for positive and negative attributes into this file
def generate_classifier(pos_usefulness, neg_usefulness):
    information = dict(sorted(pos_usefulness.items(), key=lambda item: item[1], reverse=True))
    # sort the positive information in descending order

    output_name = "nb-classifier.txt"  # the classifier data will be stored in nb-classifier.txt
    output = open(output_name, "w")
    dump_list = []  # dump_list will dump the combined model data into the classifier file

    for word in information:  # for each word in the positive attributes,
        dict1 = {word: information[word]}  # create a dictionary
        dump_list.append(dict1)  # and then append it to the list

    information = dict(sorted(neg_usefulness.items(), key=lambda item: item[1], reverse=True))

    for word in information:
        dict1 = {word: information[word]}  # for each word in the negative attributes,
        information[word] *= -1  # set the word information for each attribute as -1 for negative attributes
        dict1 = {word: information[word]}  # create a dictionary
        dump_list.append(dict1)  # and then append it to the list

    output.write(json.dumps(dump_list) + '\n')  # dump this combined model as json data into the classifier file
    output.close()


# to write the evidence values as key-value pairs into the file
def write_evidences_to_file(information, class_type):
    information = dict(sorted(information.items(), key=lambda item: item[1], reverse=True))
    # sort the information in descending order.

    output_name = class_type + "-evidences.txt"
    output = open(output_name, "w")  # create a new file for the type of evidence being stored
    dump_list = []

    for word in information:  # for each word-prob pair
        dict1 = {word: information[word]}  # store the word-prob pair as a dictionary and append that to the list
        dump_list.append(dict1)

    output.write(json.dumps(dump_list) + '\n')  # dump the list as json data into the evidence file
    output.close()


# def generate_evidence_file will generate the top_five_evidences.txt file with top 5 evidences from the dataset
def generate_evidence_file(pos_info, neg_info, N):
    pos_info = dict(sorted(pos_info.items(), key=lambda item: item[1], reverse=True))
    neg_info = dict(sorted(neg_info.items(), key=lambda item: item[1], reverse=True))
    # sort the two dictionaries in reverse order according to the mutual information value

    # open file to output the top 5 evidences of the mutual evidences generated
    output_name = "top_five_evidences.txt"
    output = open(output_name, "w")

    i = 0

    # Write top 5 positive evidences to the file
    for word in pos_info:
        if i == N:
            break
        output.write('P' + str(i + 1) + ': ' + word + '\n')
        # print(word + ': ' + str(information[word]) + '\n')
        i = i + 1

    i = 0
    output.write('\n')

    # Write top 5 negative evidences to the file
    for word in neg_info:
        if i == N:
            break
        output.write('N' + str(i + 1) + ': ' + word + '\n')
        # print(word + ': ' + str(information[word]) + '\n')
        i = i + 1

    output.close()


# read labels from the model generated using the mutual information for the purpose of preparing
# a classifier file with both the files.
def read_evidences_from_file(filename):
    with open(filename) as file:  # open the evidence file as file
        json_data = json.load(file)  # load the json data of the file, as word-probability pairs

    word_scores = {}

    # Iterating through the json list
    for word_prob_pair in json_data:  # for each word probability pair, store the word and score.
        # print(word_prob_pair)
        for word in word_prob_pair:  # select each word and score.
            word_scores[word] = word_prob_pair[word]

    # Closing file
    file.close()
    return word_scores


# print_evidences outputs the top 5 evidence for a given class type
def print_evidences(information, N, class_type):
    # sort the evidence in decreasing order by value of probability
    information = dict(sorted(information.items(), key=lambda item: item[1], reverse=True))

    print('\nTop ' + str(N) + ' evidences: \n')
    print('\n')
    i = 0

    # Print top 5 negative evidences of the given class type
    for word in information:
        if i == N:
            break
        print(class_type + str(i + 1) + ': ' + word + '\n')
        i = i + 1


# prints classification stats for the directory, detailing the percentage
# of positive and negative reviews in the directory respectively
def print_stats_for_directory(pos_filecount, neg_filecount):
    total_filecount = pos_filecount + neg_filecount
    print("\n---------------STATS FOR THIS DIRECTORY---------------\n")
    print("There are " + str(total_filecount) + " movie reviews in the given directory.\n")
    print("There are " + str(pos_filecount) + " movie reviews with positive sentiment.\n")
    print(str(round((pos_filecount / total_filecount) * 100, 2)) + "% of movie reviews had positive sentiment.\n")
    print("There are " + str(neg_filecount) + " movie reviews with negative sentiment.\n")
    print(str(round((neg_filecount / total_filecount) * 100, 2)) + "% of movie reviews had negative sentiment.\n")
    print(str(pos_filecount + neg_filecount) + " total movie reviews were classified.\n")
    print(str(total_filecount - (pos_filecount + neg_filecount)) + " movie review(s) remain unclassified.\n")
    print("------------------------------------------------------\n")


# classify_files classifies files in the given directory as either positive or negative,
# storing the results in 'results.csv'
def classify_files(directory, word_scores):
    pos_filecount = 0
    neg_filecount = 0

    pos_score = 0
    neg_score = 0

    output_name = "results.csv"
    output = open(output_name, "w")

    i = 0

    output.write('\n')

    for filename in os.listdir(directory):  # for each filename in the specified directory
        pos_score = 0
        neg_score = 0
        filepath = os.path.join(directory, filename)  # generate filepath by concatenating directory and filename
        if filepath.endswith(".txt"):  # if it is a txt file (format of files provided), then open the file
            f = open(filepath, "r")  # open the file and start pre-processing
            words = clean_text(f.read())  # remove unnecessary punctuation from the file
            for word in words:  # for each word in the file
                if word in word_scores:  # if the word is an ignore_word, then skip it
                    if word_scores[word] > 0:
                        pos_score = pos_score + word_scores[word]
                    elif word_scores[word] < 0:
                        neg_score = neg_score + word_scores[word]
            f.close()
        if abs(pos_score) > abs(neg_score):
            output.write(filename + ", 1\n")
            pos_filecount += 1
        else:
            output.write(filename + ", 0\n")
            neg_filecount += 1

    output.close()
    print_stats_for_directory(pos_filecount, neg_filecount)


# print mutual information for each attribute in the given file
def print_information(information, type):
    print("\n" + type)
    for att in information:
        print(att + ": " + str(information[att]) + "\n")


# get_sentiment is the handler method for the entire program
# get_sentiment calls the following functions:
# 1. read_evidences_from_file to read in the Naive Bayes classifier
# 2. perform_analysis to compute and print the overall sentiment of each filec in the given directory
def get_sentiment(movies_directory, classifier):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Starting sentiment analysis method...')  # Press ⌘F8 to toggle the breakpoint.

    word_scores = read_evidences_from_file(classifier)
    classify_files(movies_directory, word_scores)
    # perform_analysis(movies_directory, word_scores)
    print(f'Completed sentiment analysis. Please take a look at results.csv for the output.')
    # Press ⌘F8 to toggle the breakpoint.


# do_pre_processing is the handler method for the entire program
# do_pre_processing calls the following functions:
# 1. get_mutual_information to generate mutual information for variables in the given directories
# 2. get_useful_info to compute viability and usefulness of each attribute
# 3. generate_classifier to build a Naive Bayes classifier to compute review sentiment
def do_pre_processing(pos_directory, neg_directory):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Starting pre-processing method.')  # Press ⌘F8 to toggle the breakpoint.
    pos_info, neg_info = get_mutual_information(pos_directory, neg_directory)
    print_information(neg_info, 'NEG')

    pos_useful_info = get_useful_info(pos_info, neg_info)
    neg_useful_info = get_useful_info(neg_info, pos_info)

    write_evidences_to_file(pos_useful_info, 'pos')
    write_evidences_to_file(neg_useful_info, 'neg')
    generate_classifier(pos_useful_info, neg_useful_info)
    print(f'Done pre-processing. Please look at top_five_evidences.txt for the top 5 labels generated.')
    # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # pos_directory = input('Enter path of positive reviews:')
    # neg_directory = input('Enter path of negative reviews:')
    movies_directory = input('Enter path of movie reviews:')
    classifier = input('Enter path of Bayesian classifier:')

    # do_pre_processing(pos_directory, neg_directory)
    get_sentiment(movies_directory, classifier)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
