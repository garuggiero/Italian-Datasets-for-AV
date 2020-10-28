import os
import re
import csv
import json
import shutil
import random
import spacy
import argparse
import numpy as np
import pandas as pd
from spacy import displacy
from nltk import FreqDist
from collections import Counter
from sklearn.cross_validation import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


def parse_arguments():
    '''Read arguments from a command line'''
    parser = argparse.ArgumentParser(description='Please add arguments')
    parser.add_argument("--base_path", metavar='PATH', required=True,
        help='folder containing the tsv file with the corpus')
    parser.add_argument("--n", type=int, required=True,
        help='number of words to be considered per each author/instance')
    parser.add_argument("--gender", required=True,
        help='specify the gender setting: mixed, females, males')
    parser.add_argument("--entities", required=True,
        help='specify if data needs to contain named entities or if they are replaced: yes if entities must be present, no if entities must be replaced')
    parser.add_argument("--downsize", required=False,
        help='specify if pool of authors is downsized to the same pool of authors of the 3000 words case: it can only be yes')

    args = parser.parse_args()
    return args


def access_dataset(base_path):
    '''access the diaries and transforms them into a big pandas dataframe'''

    for (path, dirs, files) in os.walk(base_path):
        if files:
            for file in files:
                file_name = os.path.join(path, file)
                if file_name.endswith('.tsv'):
                    with open(file_name) as tsvfile:
                        if os.stat(file_name).st_size != 0:
                            data = pd.read_csv(file_name, sep='\t', names=['index','storytitle', 'author', 'gender', 'text', 'place&date', 'themes', 'period', 'emigrationcountry', 'link2story', 'link2author']) #transform it into a pandas df
                            data = data.sort_values(by='author')
                            pd.set_option('display.max_columns', None)
                            pd.set_option('display.max_rows', None)

                        elif os.stat(file_name).st_size == 0:
                            print(os.path.basename(path), ' IS EMPTY')

    return data


def get_details_topic(big_dataframe):

    '''takes the dataframe and returns three ordered lists containing the names of the authors,
    their gender and the text of each one of their stories'''

    top_authors_name = [] #top == topic
    top_authors_gender = []
    top_texts = []

    for indx, author in enumerate(big_dataframe.author):
        top_authors_name.append(author)
    for indx, gender in enumerate(big_dataframe.gender):
        top_authors_gender.append(gender)
    for indx, text in enumerate(big_dataframe.text):
        top_texts.append(text)

    top_unique_authors = set(top_authors_name)
    len_top_unique_authors = len(set(top_authors_name))

    print('There are', len_top_unique_authors, ' authors in total')
    print(' ')
    print('the blog posts are', len(top_texts), 'in total')
    zipped = zip(top_authors_name, top_texts, top_authors_gender)
    zipped = sorted(zipped)
    top_authors_name, top_texts, top_authors_gender = zip(*zipped)

    return top_authors_name, top_authors_gender, top_texts


def sort_texts(top_authors, top_texts, index_dict):
    '''takes the authors list and the list of texts and randomizes the texts from
    the same author, so that the still remain paired. It returns the randomized list of texts'''

    processed = set()
    new_top_texts = []

    for author, text in zip(top_authors, top_texts):
        if author in index_dict.keys():
            if author in processed:
                pass
            else:
                authors_texts = []
                min = index_dict[author][0]
                max = index_dict[author][1]
                max += 1

                for i in range(min, max):
                    authors_texts.append(top_texts[i])

                random.seed(42)
                random.shuffle(authors_texts)
                new_top_texts.extend(authors_texts)
                processed.add(author)

    return new_top_texts


def make_index_dict(names):
    '''returns a dictionary where the key is the name of the author and the value
     is a list indicating the min and max indx correspondent to his texts (teaken from the ordered list of texts)'''
    index_dict = dict()
    min = 0

    for indx, name in enumerate(names):
        if indx != len(names)-1 and names[indx+1] == name:
            pass
        else:
            index_dict[name] = [min, indx]
            min = indx+1
    return index_dict


def unify(names, gender, texts, index_dict):
    '''takes three ordered lists of authors' names, genders and stories and puts together the stories written by the same author in one big string.
    The NER function is also called: entities are replaced by their label.
    Returns three lists: each author name is paired with their gender and all of their stories'''

    unique_authors = []
    unified_texts = []
    all_genders = []
    analyzed = set()
    author2texts = dict()

    for indx, name in enumerate(names):
        if name in analyzed:
            pass
        else:
            checkmarks =[]
            min_indx = index_dict[name][0]
            max_index = index_dict[name][1]
            max_index += 1
            k_A = ''
            checkmark = 0

            for i in range(min_indx, max_index):
                words = word_tokenize(texts[i])
                checkmark = checkmark + (len(words)-1)
                checkmarks.append(checkmark)

                k_A += str(texts[i])
                string_length = len(k_A)+1
                k_A = k_A.ljust(string_length)

            unique_authors.append(name)
            unified_texts.append(k_A)
            all_genders.append(gender[indx])
            author2texts[name] = checkmarks
            analyzed.add(name)

    if args.entities == 'no':

        print("")
        print('Replacing entities across the documents...')

        unified_texts = find_entities(unified_texts)

    return unique_authors, unified_texts, all_genders, author2texts


def statistics(unique_names, unified_texts, all_genders, n):
    '''prints the average number of words present in the topic, the min and max number of words,
    the number of female and male authors and the total number of authors'''

    texts_lens_w = []
    count= 0

    for indx,text in enumerate(unified_texts):
        words = word_tokenize(text)
        texts_lens_w.append(len(words))

        if len(words) < n:
            count += 1

    average_len_w = sum(texts_lens_w)/len(texts_lens_w)
    min_len_w = min(texts_lens_w)
    max_len_w = max(texts_lens_w)

    print('The average len of the texts is:', average_len_w)
    print('The min number of words in the texts is', min_len_w)
    print('The max number of words in the texts is', max_len_w)
    print('')
    print(count, 'texts have less than', n, 'words')
    print('')

    gender_distr= Counter(all_genders)

    print('There are', gender_distr['Female'], 'female authors and', gender_distr['Male'],
    'male authors, and ', gender_distr['Unknown'], 'authors whose gender is not known, for a total of', len(unique_names), 'authors')


def find_entities(texts):
    '''takes a list of texts, does NER and returns a new list with replaced entities'''
    entities = []
    new_texts = []

    model_it = spacy.load('it_core_news_sm')

    for indx,text in enumerate(texts):
        doc = model_it(text)
        for entity in doc.ents:
            entities.append(entity.label_)

        #code adapted from https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
        newstring = text
        for entity in reversed(doc.ents):
            start = entity.start_char
            end = start + len(entity.text)
            if entity.label_ == 'PERSON':
                newstring = newstring[:start] + 'PERSON' + newstring[end:]
            elif entity.label_ == 'LOC':
                newstring = newstring[:start] + 'LOC' + newstring[end:]
            elif entity.label_ == 'ORG':
                newstring = newstring[:start] + 'ORG' + newstring[end:]

        new_texts.append(newstring)
    fd_ent = FreqDist(entities)

    # print('The most frequent entities are', fd_ent.most_common(20))
    # print(texts[0])
    # print(new_texts[0])

    return new_texts


def trim_3k(names, texts, genders, author2texts):
    '''
    names: list of unique author names
    texts: list of unified texts (one per author)
    genders: list of authors' gender

    return: lists of mixed authors, female and male authors with 3000 words texts
    '''

    female_3k = []
    male_3k = []

    filtered = []
    for indx, (name, text, gender) in enumerate(zip(names,texts, genders)):
        if len(word_tokenize(text)) >= 3000:
            filtered.append((name,text,gender))

    names_3k_trim, texts_3k_trim, genders_3k_trim = zip(*filtered)

    authors_with_3k_words, unknown_authors_same, known_texts, unknown_texts, known_authors_gender, to_delete = split_in_known_unknown(names_3k_trim, texts_3k_trim, genders_3k_trim, 3000, author2texts)

    for name, gender in zip(authors_with_3k_words, known_authors_gender):
        if gender == 'Female':
            female_3k.append(name)
        elif gender == 'Male':
            male_3k.append(name)

    print('The number of authors with 3000 words texts is', len(authors_with_3k_words), ' of which', len(female_3k), ' are females and ', len(male_3k), 'are males')

    return authors_with_3k_words, female_3k, male_3k


def trim(names, texts, genders, n):
    '''
    names: list of unique author names
    texts: list of unified texts (one per author)
    genders: list of author's gender
    n: min number of words per text wanted

    return: the three trimmed original lists
    '''

    print('Removing all the texts with less than', n, 'words')
    print('Removing all the authors whose gender is not known')

    less_than_n = []
    filtered = []

    for indx, (name, text, gender) in enumerate(zip(names,texts, genders)):
        if len(word_tokenize(text)) < n:
            less_than_n.append((name, indx))

        #take only texts with more than n num of words
        elif len(word_tokenize(text)) >= n and gender != 'Unknown':
            filtered.append((name,text,gender))

    names_trim, texts_trim, genders_trim = zip(*filtered)
    gender_distr = Counter(genders_trim)

    return names_trim, texts_trim, genders_trim


def downsize(names_trim, texts_trim, genders_trim, authors_with_3k_words, females_3k, males_3k):
    '''takes the trimmed author names, texts and gender lists, and the list authors with 3k words
    of the authors with 3k words texts and returns the first three lists with the same authors as in the 3k words case'''

    names_trim_downsized = []
    texts_trim_downsized = []
    genders_trim_downsized = []

    print("Downsizing number of authors...")

    if args.gender == 'mixed':
        for (name, text, gender) in (zip(names_trim, texts_trim, genders_trim)):
                for author in authors_with_3k_words:
                    if name == author:
                        names_trim_downsized.append(name)
                        texts_trim_downsized.append(text)
                        genders_trim_downsized.append(gender)
                    else:
                        pass

        print('The number of authors with 3000 words texts are ', len(authors_with_3k_words))
    else:
        female_authors, male_authors, female_texts, male_texts, female_authors_genders, male_authors_genders = extract_same_gender(names_trim, texts_trim, genders_trim)

        if args.gender == 'females':
            #downsize the current authors so that they match the female authors with 3000 words
            for (name, text, gender) in (zip(female_authors, female_texts, female_authors_genders)):
                for author in females_3k:
                    if name == author:
                        names_trim_downsized.append(name)
                        texts_trim_downsized.append(text)
                        genders_trim_downsized.append(gender)
                    else:
                        pass
            print('The number of female authors with 3000 words texts are ', len(females_3k))

        elif args.gender == 'males':
            for (name, text, gender) in (zip(male_authors, male_texts, male_authors_genders)):
                for author in males_3k:
                    if name == author:
                        names_trim_downsized.append(name)
                        texts_trim_downsized.append(text)
                        genders_trim_downsized.append(gender)
                    else:
                        pass

            print('The number of male authors with 3000 words texts are ', len(males_3k))

    names_trim_downsized, texts_trim_downsized, genders_trim_downsized = shuffle(names_trim_downsized, texts_trim_downsized, genders_trim_downsized)
    print('When downsizing, the total number of authors becomes ', len(names_trim_downsized))


    return names_trim_downsized, texts_trim_downsized, genders_trim_downsized


def shuffle(names_trim, texts_trim, genders_trim):
    '''takes three lists and shuffles them'''

    print('Shuffling...')
    zipped = list(zip(names_trim, texts_trim, genders_trim))
    random.seed(42)
    random.shuffle(zipped)
    names_trim, texts_trim, genders_trim = zip(*zipped)

    return names_trim, texts_trim, genders_trim


def extract_same_gender(trimmed_authors, trimmed_texts, trimmed_genders):
    '''takes the three trimmed lists of authors, texts and genders and a number n
    and separates them into female and male authors, texts and gender lists'''

    female_authors = []
    male_authors = []
    female_texts = []
    male_texts = []
    female_authors_genders = []
    male_authors_genders = []

    for (author, text, gender) in zip(trimmed_authors, trimmed_texts, trimmed_genders):
        if gender == 'Female':
            female_authors.append(author)
            female_texts.append(text)
            female_authors_genders.append(gender)

        elif gender == 'Male':
            male_authors.append(author)
            male_texts.append(text)
            male_authors_genders.append(gender)

    return female_authors, male_authors, female_texts, male_texts, female_authors_genders, male_authors_genders


def split_in_known_unknown(trimmed_authors, trimmed_texts, trimmed_genders, n, author2texts):
    '''takes the three lists of authors, texts and genders and a number of words n
    and returns:

    a list of known Authors
    a list of unknown Authors
    a list of known texts
    a list of unknown texts
    a list of known authors genders
    a list of unknown authors genders
    a list of deleted authors
    '''

    known_authors = []
    known_authors_gender = []
    known_texts = []
    unknown_texts = []

    to_delete = []

    for (author, text, gender) in zip(trimmed_authors, trimmed_texts, trimmed_genders):
        words = word_tokenize(text)
        checkmarks = author2texts[author]
        known_t = []
        unknown_t = []

        start_unknown = None

        for indx, checkmark in enumerate(checkmarks):
            if len(known_t) != int(n/2):
                if checkmark < int(n/2):
                    pass
                elif checkmark >= int(n/2):
                    if checkmark != checkmarks[-1]:
                        start_unknown = checkmarks[indx+1]
                        known_t = [word for word in words[:int(n/2)]]

                    else:
                        to_delete.append([author,text,gender])


            else:
                if len(unknown_t) != int(n/2):
                    diff = checkmark - start_unknown
                    if diff < int(n/2):
                        if checkmark == checkmarks[-1]:
                            to_delete.append([author,text,gender])
                        else:
                            pass
                    elif diff == int(n/2):
                        unknown_t = [word for word in words[start_unknown:checkmark]]

                    elif diff > int(n/2):
                        unknown_t = [word for word in words[start_unknown:]]
                        unknown_t = [word for word in unknown_t[:int(n/2)]]


        if len(known_t) != 0 and len(unknown_t) != 0:
            known = ' '.join(known_t)
            unknown = ' '.join(unknown_t)

            known_authors.append(author)
            known_texts.append(known)
            unknown_texts.append(unknown)
            known_authors_gender.append(gender)

    print('In the process ', str(len(trimmed_authors) - len(known_authors)),' authors have been deleted')

    gender_distr= Counter(known_authors_gender)

    print('In the new subset there are', gender_distr['Female'], 'female authors and', gender_distr['Male'],
    'male authors, for a total of', len(known_authors), 'authors')

    unknown_authors_same = [author for author in known_authors]


    return known_authors, unknown_authors_same, known_texts, unknown_texts, known_authors_gender, to_delete


def make_known_unknown(known_authors, unknown_authors_same, known_texts, unknown_texts, known_authors_gender):
    ''''takes lists of known Authors, unknown Authors, known texts, unknown texts, known authors genders and unknown authors genders
    and returns:
    a dictionary {'author': [number, known_text, unknown_text, gender]}
    known_unknown pairs to be used for training and testing'''

    author2info = dict()

    unknown_authors_gender_same = [gender for gender in known_authors_gender]

    number = 1
    for indx, author in enumerate(known_authors):
        author2info[author] = [number, known_texts[indx], unknown_texts[indx], known_authors_gender[indx]]
        number += 1

    # print(author2index)
    yes_label = 'Y'
    no_label = 'N'
    k_u_pairs = []

    for indx, (k_A, u_A, k_T, u_T) in enumerate(zip(known_authors, unknown_authors_same, known_texts, unknown_texts)):
        if indx <= round((len(known_authors)-1)/2): #from index 0 to half (12)
            k_u_pairs.append([(k_T, u_T), yes_label, author2info[k_A][3], author2info[u_A][3], k_A, u_A])

        elif indx == round(len(known_authors)/2): #index half+1 (13)
            k_u_pairs.append([(k_T, unknown_texts[len(known_authors)-1]), no_label, author2info[k_A][3], unknown_authors_gender_same[len(known_authors)-1], k_A, unknown_authors_same[len(known_authors)-1]])

        elif indx == len(known_authors)-1: #last index
            k_u_pairs.append([(k_T, unknown_texts[len(known_authors)-2]), no_label, author2info[k_A][3], unknown_authors_gender_same[len(known_authors)-2], k_A, unknown_authors_same[len(known_authors)-2]])

        else: #index in range half+1, last-1
            k_u_pairs.append([(k_T, unknown_texts[indx+1]), no_label, author2info[k_A][3], unknown_authors_gender_same[indx+1], k_A, unknown_authors_same[indx+1]])



    return author2info, k_u_pairs


def train_test(known_unknown_pairs):
    '''takes the known_unknown pairs and splits them into training and test'''

    k_u_tuples = []
    labels = []
    k_u_gender = dict()
    k_u_authors = dict()


    for elem in known_unknown_pairs:
        k_u_tuples.append(elem[0])
        labels.append(elem[1])
        k_u_gender[elem[0]] = [elem[2], elem[3]]
        k_u_authors[elem[0]] = [elem[4], elem[5]]


    X_train, X_test, y_train, y_test = train_test_split(k_u_tuples, labels, test_size=0.30, random_state=42)

    print('')
    print(len(X_train), 'pairs are used for training', len(X_test), 'pairs are used for testing')

    train_k_u_authors = []
    test_k_u_authors = []

    for elem in X_train:
        train_k_u_authors.append(k_u_authors[elem])
    for elem in X_test:
        test_k_u_authors.append(k_u_authors[elem])

    train_k_gender = []
    train_u_gender = []

    test_k_gender = []
    test_u_gender = []

    for elem in X_train:
        train_k_gender.append(k_u_gender[elem][0])
        train_u_gender.append(k_u_gender[elem][1])

    for elem in X_test:
        test_k_gender.append(k_u_gender[elem][0])
        test_u_gender.append(k_u_gender[elem][1])

    print('Gender distr known authors in train:', Counter(train_k_gender))
    print('Gender distr UNknown authors in train:', Counter(train_u_gender))

    print('Gender distr known authors in test:', Counter(test_k_gender))
    print('Gender distr UNknown authors in test:', Counter(test_u_gender))


    return X_train, X_test, y_train, y_test, k_u_gender, train_k_gender, train_u_gender, test_k_gender, test_u_gender, train_k_u_authors, test_k_u_authors


def make_folders(X, y, specify_set, specify_n, specify_gender):

        root_path = '/home/gaetana/Desktop/AV_Diaries/'

        folders = []

        for i in range(1, len(X)+1):
            if i <= 9:
                folder_name = 'IT00'
                folder_name += str(i)
                folders.append(folder_name)
            elif i > 9:
                folder_name = 'IT0'
                folder_name += str(i)
                folders.append(folder_name)
            else:
                folder_name = 'IT'
                folder_name += str(i)
                folders.append(folder_name)

        if args.downsize == 'yes':
            if os.path.exists(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'):
                shutil.rmtree(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized')

            os.mkdir(os.path.join(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'))
            processed = set()

            with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/contents.json', mode='w', encoding='utf-8') as f:
                entry = {"language": "Italian", "problems": folders}
                json.dump(entry, f)


            for folder in folders:
                os.mkdir(os.path.join(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/')+str(folder))

                for indx,(elem, label) in enumerate(zip(X, y)):
                    if indx in processed:
                        pass
                    else:
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/'+str(folder)+'/known01.txt', 'w') as known:
                            known.write(elem[0])
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/'+str(folder)+'/unknown.txt', 'w') as unknown:
                            unknown.write(elem[1])
                            processed.add(indx)
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/truth.txt', 'a') as truth:
                            truth.write(folder+' '+label+'\n')
                            break
        else:

            if os.path.exists(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender):
                shutil.rmtree(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender)

            os.mkdir(os.path.join(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender))
            processed = set()

            with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/contents.json', mode='w', encoding='utf-8') as f:
                entry = {"language": "Italian", "problems": folders}
                json.dump(entry, f)


            for folder in folders:
                os.mkdir(os.path.join(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/')+str(folder))

                for indx,(elem, label) in enumerate(zip(X, y)):
                    if indx in processed:
                        pass
                    else:
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/'+str(folder)+'/known01.txt', 'w') as known:
                            known.write(elem[0])
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/'+str(folder)+'/unknown.txt', 'w') as unknown:
                            unknown.write(elem[1])
                            processed.add(indx)
                        with open(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/truth.txt', 'a') as truth:
                            truth.write(folder+' '+label+'\n')
                            break
        return folders

def make_info_file(folder_names, X, y, k_gender, u_gender, specify_set, specify_n, specify_gender):
    '''make an info file containing foldern name, known_unknown text pairs, gold label, known&unknown gender'''

    root_path = '/home/gaetana/Desktop/AV_Diaries/'

    if args.downsize == 'yes':
            if os.path.exists(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/INFO.csv'):
                os.remove(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/INFO.csv')
                print("File Removed!")

            dictionary = {'problem ID': folder_names, 'known_unknown pairs': X, 'k_gender': k_gender, 'u_gender': u_gender, 'gold label': y}
            df = pd.DataFrame(data=dictionary)

            csv_file = df.to_csv(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'_downsized'+'/INFO.csv', encoding='utf-8', index=False)
    else:
        if os.path.exists(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/INFO.csv'):
            os.remove(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/INFO.csv')
            print("File Removed!")


        dictionary = {'problem ID': folder_names, 'known_unknown pairs': X, 'k_gender': k_gender, 'u_gender': u_gender, 'gold label': y}
        df = pd.DataFrame(data=dictionary)

        csv_file = df.to_csv(root_path+'diaries_'+specify_set+'_'+specify_n+'_'+specify_gender+'/INFO.csv', encoding='utf-8', index=False)

    return df

#####################################################
def main():

    print('Accessing the dataset and making the dataframe...')
    dataframe = access_dataset(args.base_path)
    print('')
    print('Extracting the details...')

    top_authors_name, top_authors_gender, top_texts = get_details_topic(dataframe)

    index_dict = make_index_dict(top_authors_name)
    new_top_texts = sort_texts(top_authors_name, top_texts, index_dict)

    print('')
    print('Unifying the information...')
    unique_authors, unified_texts, all_genders, author2texts = unify(top_authors_name, top_authors_gender, new_top_texts, index_dict)

    print('')
    print('Before trimming------')
    print('')

    n = args.n

    statistics(unique_authors, unified_texts, all_genders, n)

    print('')
    print('After trimming------')
    unique_authors_trim, unified_texts_trim, all_genders_trim= trim(unique_authors, unified_texts, all_genders, n)

    if args.downsize == 'yes':
        authors_3k_words, females_3k, males_3k = trim_3k(unique_authors, unified_texts, all_genders, author2texts)

        if args.gender == 'mixed':
            unique_authors_trim, unified_texts_trim, all_genders_trim = downsize(unique_authors_trim, unified_texts_trim, all_genders_trim, authors_3k_words, females_3k, males_3k)
        elif args.gender == 'females':
            female_authors, female_texts, female_authors_genders = downsize(unique_authors_trim, unified_texts_trim, all_genders_trim, authors_3k_words, females_3k, males_3k)
        elif args.gender == 'males':
            male_authors, male_texts, male_authors_genders = downsize(unique_authors_trim, unified_texts_trim, all_genders_trim, authors_3k_words, females_3k, males_3k)

    else:
        print('')
        unique_authors_trim, unified_texts_trim, all_genders_trim = shuffle(unique_authors_trim, unified_texts_trim, all_genders_trim)


    print('')

    ######### MIXED GENDER
    if args.gender == 'mixed':
        print('')
        # statistics(unique_authors_trim, unified_texts_trim, all_genders_trim, n)
        known_authors, unknown_authors_same, known_texts, unknown_texts, known_authors_gender, to_delete = split_in_known_unknown(unique_authors_trim, unified_texts_trim, all_genders_trim, n, author2texts)
        author2info, k_u_pairs = make_known_unknown(known_authors, unknown_authors_same, known_texts, unknown_texts, known_authors_gender)

        print('Splitting into training and test...')

        X_train, X_test, y_train, y_test, k_u_gender, train_k_gender, train_u_gender, test_k_gender, test_u_gender, train_k_u_authors, test_k_u_authors =  train_test(k_u_pairs)


    ######### SEPARATE GENDER

    elif args.gender != 'mixed':
        if args.downsize == 'yes':
            pass
        else:
            female_authors, male_authors, female_texts, male_texts, female_authors_genders, male_authors_genders = extract_same_gender(unique_authors_trim, unified_texts_trim, all_genders_trim)


        ######## FEMALES
        if args.gender == 'females':

            print('Extracting only female authors and their texts...')
            print('')
            statistics(female_authors, female_texts, female_authors_genders, n)

            print('')
            print('Creating the known-unknown female pairs...')
            print('')
            known_authors_female, unknown_authors_female_same, known_texts_female, unknown_texts_female, female_authors_gender, to_delete = split_in_known_unknown(female_authors, female_texts, female_authors_genders, n, author2texts)
            author2info_female, k_u_pairs_female = make_known_unknown(known_authors_female, unknown_authors_female_same, known_texts_female, unknown_texts_female, female_authors_gender)
            print('')


            print('Splitting into training and test...')
            X_train, X_test, y_train, y_test, k_u_gender, train_k_gender, train_u_gender, test_k_gender, test_u_gender, train_k_u_authors, test_k_u_authors =  train_test(k_u_pairs_female)


        ######## MALES
        elif args.gender == 'males':

            print('Extracting only male authors and their texts')
            # statistics(male_authors, male_texts, male_authors_genders, n)

            print('')
            print('Creating the known-unknown male pairs')
            known_authors_male, unknown_authors_male_same, known_texts_male, unknown_texts_male, male_authors_genders, to_delete = split_in_known_unknown(male_authors, male_texts, male_authors_genders, n, author2texts)
            author2info_male, k_u_pairs_male = make_known_unknown(known_authors_male, unknown_authors_male_same, known_texts_male, unknown_texts_male, male_authors_genders)
            print('')

            print('Splitting into training and test...')
            X_train, X_test, y_train, y_test, k_u_gender, train_k_gender, train_u_gender, test_k_gender, test_u_gender, train_k_u_authors, test_k_u_authors =  train_test(k_u_pairs_male)

    folders_train = make_folders(X_train, y_train, 'train', str(args.n), str(args.gender))
    folders_test = make_folders(X_test, y_test, 'test', str(args.n), str(args.gender))

    df_train = make_info_file(folders_train, X_train, y_train, train_k_gender, train_u_gender, 'train', str(args.n), str(args.gender))
    df_test = make_info_file(folders_test, X_test, y_test, test_k_gender, test_u_gender, 'test', str(args.n), str(args.gender))




if __name__ == '__main__':
    args = parse_arguments()
    main()
