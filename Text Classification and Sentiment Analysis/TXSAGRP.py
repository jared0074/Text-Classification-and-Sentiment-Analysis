# ##Question 1.3
# import nltk
# from textblob import TextBlob
# from nltk.util import pad_sequence
# from nltk.util import bigrams
# from nltk.util import ngrams
# from nltk.util import everygrams
# from collections import Counter
# from nltk.corpus import reuters
# from nltk.lm.preprocessing import pad_both_ends
# from nltk.lm.preprocessing import flatten
# from nltk.util import pad_sequence
# from nltk.util import everygrams
# from nltk.lm.preprocessing import padded_everygram_pipeline
# from nltk.lm import MLE
# from nltk.lm import Laplace
# from nltk.lm.api import LanguageModel, Smoothing
#
# def flatten(A):
#     rt = []
#     for i in A:
#         if isinstance(i,list): rt.extend(flatten(i))
#         else: rt.append(i)
#     return rt
#
# #read data form file
# group_data = open("C:/Users/jared/Desktop/Text Corpus.txt","r")
# txt_data = group_data.read()
#
# stripped1_text = ""
#
# #remove <s> and </s>
# txt_data = txt_data.replace("<s>","")
# txt_data = txt_data.replace("</s>","")
# for i in txt_data:
#     stripped1_text += i.strip("\n")
#
# tokens = nltk.tokenize.word_tokenize(stripped1_text.lower())
#
# print (Counter(tokens))
# #bigrams = ngrams(tokenize_list,2)
# #print(Counter(bigrams))
#
#
# tokens = list(everygrams(tokens, max_len=1))
# n = 1
# train_data, padded_sents = padded_everygram_pipeline(n, tokens)
# smoothed_model = Laplace(n)
# smoothed_model.fit(train_data, padded_sents)
#
# print("Smoothed")
# print("P(he)        : ",smoothed_model.score('he'))
# print("P(read)      : ",smoothed_model.score('read'))
# print("P(a)         : ",smoothed_model.score('a'))
# print("P(book)      : ",smoothed_model.score('book'))
# print("P(i)         : ",smoothed_model.score('i'))
# print("P(different) : ",smoothed_model.score('different'))
# print("P(my)        : ",smoothed_model.score('my'))
# print("P(mulan)     : ",smoothed_model.score('mulan'))
# tokens = list(everygrams(tokens, max_len=1))
# n = 1
# train_data, padded_sents = padded_everygram_pipeline(n,tokens)
# model = MLE(n)
# model.fit(train_data,padded_sents)
#
# print("Unsmoothed")
# print("P(he)        : ",model.score('he'))
# print("P(read)      : ",model.score('read'))
# print("P(a)         : ",model.score('a'))
# print("P(book)      : ",model.score('book'))
# print("P(i)         : ",model.score('i'))
# print("P(different) : ",model.score('different'))
# print("P(my)        : ",model.score('my'))
# print("P(mulan)     : ",model.score('mulan'))


##Question 2.3
# import nltk
# from textblob import TextBlob
# from nltk.util import pad_sequence
# from nltk.util import bigrams
# from nltk.util import ngrams
# from nltk.util import everygrams
# from collections import Counter
# from nltk.corpus import reuters
# from nltk.lm.preprocessing import pad_both_ends
# from nltk.lm.preprocessing import flatten
# from nltk.util import pad_sequence
# from nltk.util import everygrams
# from nltk.lm.preprocessing import padded_everygram_pipeline
# from nltk.lm import MLE
# from nltk.lm import Laplace
# from nltk.lm.api import LanguageModel, Smoothing
#
# def flatten(A):
#     rt = []
#     for i in A:
#         if isinstance(i,list): rt.extend(flatten(i))
#         else: rt.append(i)
#     return rt
#
# #read data form file
# group_data = open("Text Corpus.txt","r")
# txt_data = group_data.read()
#
# stripped1_text = ""
#
# #remove <s> and </s>
# txt_data = txt_data.replace("<s>","")
# txt_data = txt_data.replace("</s>","")
# for i in txt_data:
#     stripped1_text += i.strip("\n")
#
# tokens = nltk.tokenize.word_tokenize(stripped1_text.lower())
# tokens = list(everygrams(tokens, max_len = 2))
# n = 2
# train_data, padded_sents = padded_everygram_pipeline(n,tokens)
# model = MLE(n)
# model.fit(train_data,padded_sents)
#
# print("Unsmoothed Bigram")
# print("P(He|<s>)        :",model.score('he','<s>'.split()))
# print("P(I|<s>)         :",model.score('i','<s>'.split()))
# print("P(read|He)       :",model.score('read','he'.split()))
# print("P(read|I)        :",model.score('read','i'.split()))
# print("P(a|read)        :",model.score('a','read'.split()))
# print("P(book|a)        :",model.score('book','a'.split()))
# print("P(different|a)   :",model.score('different','a'.split()))
# print("P(book|different):",model.score('book','different'.split()))
# print("P(</s>|book)     :",model.score('</s>','book'.split()))
# print("P(my|book)       :",model.score('my','book'.split()))
# print("P(Mulan|my)      :",model.score('mulan','my'.split()))
# print("P(</s>|Mulan)    :",model.score('</s>','mulan'.split()))
#
# #read data form file
# group_data = open("Text Corpus.txt","r")
# txt_data = group_data.read()
#
# stripped1_text = ""
#
# #remove <s> and </s>
# txt_data = txt_data.replace("<s>","")
# txt_data = txt_data.replace("</s>","")
# for i in txt_data:
#     stripped1_text += i.strip("\n")
#
# tokens = nltk.tokenize.word_tokenize(stripped1_text.lower())
# tokens = list(everygrams(tokens, max_len = 2))
# n = 2
# train_data, padded_sents = padded_everygram_pipeline(n,tokens)
# smoothed_model = Laplace(n)
# smoothed_model.fit(train_data,padded_sents)
#
# print("Lapalce Smoothed Bigram")
# print("P(He|<s>)        :",smoothed_model.score('he','<s>'.split()))
# print("P(I|<s>)         :",smoothed_model.score('i','<s>'.split()))
# print("P(read|He)       :",smoothed_model.score('read','he'.split()))
# print("P(read|I)        :",smoothed_model.score('read','i'.split()))
# print("P(a|read)        :",smoothed_model.score('a','read'.split()))
# print("P(book|a)        :",smoothed_model.score('book','a'.split()))
# print("P(different|a)   :",smoothed_model.score('different','a'.split()))
# print("P(book|different):",smoothed_model.score('book','different'.split()))
# print("P(</s>|book)     :",smoothed_model.score('</s>','book'.split()))
# print("P(my|book)       :",smoothed_model.score('my','book'.split()))
# print("P(Mulan|my)      :",smoothed_model.score('mulan','my'.split()))
# print("P(</s>|Mulan)    :",smoothed_model.score('</s>','mulan'.split()))


# ##Question 3.4
# ##Unigram Model
# # pHe = smoothed_model.score('he')
# # pRead = smoothed_model.score('read')
# # pA = smoothed_model.score('a')
# # pBook = smoothed_model.score('book')
# # pI = smoothed_model.score('i')
# # pDifferent = smoothed_model.score('different')
# # pMy = smoothed_model.score('my')
# # pMulan = smoothed_model.score('mulan')
#
# # pSHeReadABookS = pHeS * pReadHe * pARead * pBookA * pSBook
# # pSIReadADifferentBookS = pIS * pReadI * pARead * pDifferentA * pBookDifferent * pSBook
# # pSHeReadABookMyMulanS = pHeS * pReadHe * pARead * pBookA * pMyBook * pMulanMy * pSMulan
#
# # print("-- Sentence Probabilities using Smoothed Unigram Language Model")
# # print("The sentence probabilty of 'He read a book' is", format(float(pSHeReadABookS), '.20f'))
# # print("The sentence probabilty of 'He read a book' is", format(float(pSIReadADifferentBookS), '.20f'))
# # print("The sentence probabilty of 'He read a book' is", format(float(pSHeReadABookMyMulanS), '.20f'))
#
# ##Bigram Model
# pHeS = smoothed_model.score('he','<s>'.split())
# pReadHe = smoothed_model.score('read','he'.split())
# pARead = smoothed_model.score('a','read'.split())
# pBookA = smoothed_model.score('book','a'.split())
# pSBook = smoothed_model.score('</s>','book'.split())
# pIS = smoothed_model.score('i','<s>'.split())
# pReadI = smoothed_model.score('read','i'.split())
# pDifferentA = smoothed_model.score('different','a'.split())
# pBookDifferent = smoothed_model.score('book','different'.split())
# pMyBook = smoothed_model.score('my','book'.split())
# pMulanMy = smoothed_model.score('mulan','my'.split())
# pSMulan = smoothed_model.score('</s>','mulan'.split())
#
# pSHeReadABookS = pHeS * pReadHe * pARead * pBookA * pSBook
# pSIReadADifferentBookS = pIS * pReadI * pARead * pDifferentA * pBookDifferent * pSBook
# pSHeReadABookMyMulanS = pHeS * pReadHe * pARead * pBookA * pMyBook * pMulanMy * pSMulan
#
# print("-- Sentence Probabilities using Smoothed Unigram Language Model")
# print("The sentence probabilty of 'He read a book' is", format(float(pSHeReadABookS), '.20f'))
# print("The sentence probabilty of 'He read a book' is", format(float(pSIReadADifferentBookS), '.20f'))
# print("The sentence probabilty of 'He read a book' is", format(float(pSHeReadABookMyMulanS), '.20f'))
#
# group_data.close()

# ##Question 4.1
# from textblob import TextBlob
# import pandas as pd
#
# data = pd.read_csv("C:/Users/benhu/OneDrive/Desktop/TXSA/Assignment-20201215/Group Assignment Data/Musical_Instruments_Reviews.csv")
# data.head()
# reviews = data['Reviews']
#
# for line in reviews:
#     print(line)
#     sent = TextBlob(line)
#     print("The polarity is: ", sent.polarity)
#     if (sent.polarity == 0) :
#         print("The sentiment of the sentence is neutral")
#     else:
#         if (sent.polarity > -0) :
#             print("The sentiment of the sentence is positive")
#         else:
#             print("The sentiment of the sentence is negative")
#         print()


##Question 4.2
# from textblob import TextBlob
# import pandas as pd

# csv_file = ("C:/Users/benhu/OneDrive/Desktop/TXSA/Assignment-20201215/Group Assignment Data/Musical_Instruments_Reviews.csv")
# data = pd.read_csv(csv_file)
# reviews = data['Reviews']
#
# #Create list to store polarity and sentiment
# polarityRow = []
# sentimentRow = []
#
# for line in reviews:
#     print(line)
#     sent = TextBlob(line)
#     sentiment = ""
#     print("The polarity is: ", sent.polarity)
#     if (sent.polarity == 0) :
#         print("The sentiment of the sentence is neutral")
#         sentiment = "Neutral"
#     else:
#         if (sent.polarity > -0) :
#             print("The sentiment of the sentence is positive")
#             sentiment = "Positive"
#         else:
#             print("The sentiment of the sentence is negative")
#             sentiment = "Negative"
#         print()
#
#     # Add polarity and sentiment into the list created
#     polarityRow.append(sent.polarity)
#     sentimentRow.append(sentiment)
#
# #Create new column to store polarity and sentiment
# data["Polarity"] = polarityRow
# data["Sentiment"] = sentimentRow
#
# #Save the changes in CSV file
# data.to_csv(csv_file, index=False)


# ##Question 4.3
# import pandas as pd
# import nltk.corpus as cp
# from nltk.tokenize import word_tokenize
# import random
# import collections
#
# #Feature extractor function
# def wordFeatures(word):
#     stopset =list(set(cp.stopwords.words('english')))
#     return dict([(word,True) for word in word_tokenize(word) if word not in stopset])
#
# #Read file
# file = pd.read_csv("Musical_Instruments_Reviews.csv")
# #Process data so that 'Review' column and 'Sentiment' column is taken
# data = [(wordFeatures(row['Reviews']),row['Sentiment'])for n, row in file.iterrows()]
#
# #Shuffling the data
# random.shuffle(data)
#
# #Splitting data into train and test datasets
# split = round(len(data)*0.7)
# train_set,test_set = data[:split], data[split:]
#
# #Creating the NaiveBayes classifier
# classifier = nltk.NaiveBayesClassifier.train(train_set)
#
# #two testsets, one is the original test set
# #the other is the one to be predicted by classifier
# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)
#
# #adding the predictions of model into testset
# for i, (feats, label) in enumerate(test_set):
#     refsets[label].add(i)
#     observed = classifier.classify(feats)
#     testsets[observed].add(i)
#
# # Show the accuracy of the Naive Bayes Classifier
# print("Accuracy of Naive Bayes Classifier:",  nltk.classify.accuracy(classifier, test_set) )
# # Show average performance measures of the model
# print("\nAverage of Performance measures:\n")
# print("Precision : ", (nltk.scores.precision(refsets['Negative'], testsets['Negative']) + nltk.scores.precision(refsets['Neutral'] , testsets['Neutral']) + nltk.scores.precision(refsets['Positive'], testsets['Positive']))/3)
# print("Recall    : ", (nltk.scores.recall(refsets['Negative'], testsets['Negative']) + nltk.scores.recall(refsets['Neutral'] , testsets['Neutral']) + nltk.scores.recall(refsets['Positive'], testsets['Positive']))/3)
# print("F1-score  : ", (nltk.scores.f_measure(refsets['Negative'], testsets['Negative']) + nltk.scores.f_measure(refsets['Neutral'] , testsets['Neutral']) + nltk.scores.f_measure(refsets['Positive'], testsets['Positive']))/3)
#
# # Show performance measures for each sentiment
# print("\nPerformance measures for each sentiment:\n")
# print("\t\tprecision\trecall\t\tF1-score")
# print("Negative\t" , "%.2f" % nltk.scores.precision(refsets['Negative'], testsets['Negative']),"\t\t","%.2f" % nltk.scores.recall(refsets['Negative'], testsets['Negative']),"\t\t", "%.2f" % nltk.scores.f_measure(refsets['Negative'], testsets['Negative']),"\t")
# print("Neutral\t\t", "%.2f" % nltk.scores.precision(refsets['Neutral'] , testsets['Neutral']) ,"\t\t","%.2f" % nltk.scores.recall(refsets['Neutral'] , testsets['Neutral']) ,"\t\t", "%.2f" % nltk.scores.f_measure(refsets['Neutral'], testsets['Neutral'])  ,"\t")
# print("Positive\t" , "%.2f" % nltk.scores.precision(refsets['Positive'], testsets['Positive']),"\t\t","%.2f" % nltk.scores.recall(refsets['Positive'], testsets['Positive']),"\t\t", "%.2f" % nltk.scores.f_measure(refsets['Positive'], testsets['Positive']),"\t")
