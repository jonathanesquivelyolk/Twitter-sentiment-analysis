# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:10:27 2018

@author: jonat
"""

import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
import random
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet 
import re

def randomExclusiveArray(size,a,b):
    arr = [random.uniform(a,b) for x in range(size)]
    return np.array(arr).astype('float32')

def decision(probability):
    return random.random() < probability

class semEvalData:
    
    def __init__(self,data_path,n_dims=25):
        
        
        self.n_dims = n_dims
        
        #To remember our vocabulary:
        self.words_in_corpus = {}
        self.maxLength=0
        
        data=[]
        with open(data_path,"r",encoding="utf-8-sig") as inFile:
            for row in inFile:
                row=row[:-1] #trim newline
                ndx,tweet,c = row.split("\t")  #splitTokens
                tokens = tknzr.tokenize(tweet)
                numTokens = len(tokens)
                data.append(list([ndx,tweet,numTokens,c]))
                for tok in tokens:
                    if tok not in self.words_in_corpus:
                        self.words_in_corpus[tok]=1
                    else:
                        self.words_in_corpus[tok]+=1
                if len(tokens) > self.maxLength:
                    self.maxLength = len(tokens)
                    
        self.df = pd.DataFrame(data,columns=["ndx","Tweet","numTokens","Sentiment"])
        
        
    def exploreRealData(self):
        return self.df
    
    #pd.get_dummies(y_df)
    def grab_data(self,percentTraining=.80,validation=False,percentValidation=.1):
        cnt=0
        try_again1 = True
        
        #Split train and test
        while try_again1 and cnt<20:
            trainVal_df, test_df  = train_test_split(self.df[["ndx","Tweet","Sentiment"]],
                                                  test_size = 1-percentTraining)
            
            try_again1 = False
            if len(trainVal_df["Sentiment"].unique()) < len(test_df["Sentiment"].unique()):
                #our training data is missing some classes...
                print("Bad split, splitting again...")
                try_again1 = True
                cnt+=1                  
        
        try_again1 = True
        cnt=0
        percentValidation = percentValidation / percentTraining
        #split train and validation;
        while try_again1 and cnt<20:
            train_df, val_df = train_test_split(trainVal_df, test_size = percentValidation)
            try_again1 = False
            if len(val_df["Sentiment"].unique()) < len(train_df["Sentiment"].unique()):
                #our training data is missing some classes...
                print("Bad split, splitting again...")
                try_again1 = True
                cnt+=1               
        return train_df,test_df,val_df
                
                
                
    def load_embeddings(self,embPath):
        
        #dictionary of embeddings
        self.embDict={0:np.zeros(self.n_dims)}
        
        #dicitonary of words to ids
        self.word2id={"<pad>":0}
        
        #ditionary of id to words
        self.id2word={0:"<pad>"}
        
        #0 shall be resevred for padding
        cnt_id=1
         
        allcnt=0
        with open(embPath,"r",encoding="utf-8") as embFile:
            for line in embFile:
                tokens = line.split()
                
                #only add to our dictionary if it is in our corpus
                if tokens[0] in self.words_in_corpus.keys():
                    self.word2id[tokens[0]]=int(cnt_id)
                    self.id2word[int(cnt_id)]=tokens[0]
                    self.embDict[int(cnt_id)]=np.array(tokens[1:]).astype('float32')
                    cnt_id+=1
                
                allcnt+=1
                
        #Now assign ids to words in corpus not in embeddings
        missingCount = 0
        for word in self.words_in_corpus.keys():
            if word not in self.word2id:
                self.word2id[word]=int(cnt_id)
                self.id2word[int(cnt_id)]=word
                
                #use random embeddings for unkown tokens
                self.embDict[int(cnt_id)]=randomExclusiveArray(self.n_dims,-1,1)
                cnt_id+=1
                missingCount+=1
                
        return self.embDict,allcnt,missingCount
    
    
    def tokenize_data(self,X_put,pad=True,padLength=-1):
        if padLength < 0:
            padLength = self.maxLength
        data=[]
        #A function that will tokenize each tweet into its Ids
        for tweet in X_put:
            tokens = tknzr.tokenize(tweet)
            tweet_tokens = [self.word2id[tok] for tok in tokens]
            if pad:
                if len(tweet_tokens) < padLength:
                    desired = np.zeros(padLength,dtype="int32")
                    desired[:len(tweet_tokens)]=tweet_tokens
                elif padLength <= len(tweet_tokens):
                    desired = np.array(tweet_tokens[:padLength],dtype="int32")
                data.append(desired)
        
        return np.array(data,dtype="int32")
        
    def getEmbeddingTable(self):
        return [self.embDict[word_id] for word_id in range(len(self.word2id))]


    def combClasses(self,df):
        aug_data=[]
        for row in df[["ndx","Tweet","Sentiment"]].values:
            old_ndx,tweet,c = row[0], row[1], row[2]
            
            # Modify the other classes into one big class
            if c == "objective" or c=="objective-OR-neutral":
                c = "neutral"
                row = np.array([old_ndx,tweet,c])
                
            aug_data.append(row) 
        aug_df = pd.DataFrame(data=aug_data,columns=["ndx","Tweet","Sentiment"])
        return aug_df
        
    #numAugs will reflect how many times we augment a single datapoint
    #we will support up to N many.
    def augmentData(self,train_df,numAugs=2,giveReplacements=False,probKeepPositive=1,probKeepNeutral=1):
        #This function is designed to make augmenation on tweets by genearting synonyms of words
        #we then will replace the word with a randomly chosen synonym from that set
        #if the set is null we will try again with a different index
        #If we fail after 5 tries we will skip it.
        def getSyns(word):
            synonyms = [] 
            for syn in wordnet.synsets(word):
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
                    
            #ensure the new word is proccessed correctly
            ret=[]
            #check for same word            
            #check for number or puncuation
            for syn in synonyms:
                if re.match("^[a-zA-Z]*$", syn):
                    #if it is a captial (for some reason...) it is likely a name we should get rid of it
                    if syn == syn.lower():
                        #make sure we do not include the word
                        if syn != word:
                            ret.append(syn)
            
            return set(ret)

        replacedWords=[]
        posCnt=0
        posSkipcnt=0
        aug_data=[]
        for row in train_df[["ndx","Tweet","Sentiment"]].values:
            old_ndx,tweet,c = row[0], row[1], row[2]
            

            
            #tokenize the tweets
            tokens = tknzr.tokenize(tweet)
            og_tokens = tokens
            #get a new augmentation for the 
            
            #chose random indexes to augment:
            all_index = [i for i in range(len(tokens))]
            aug_indexs = random.sample(range(len(tokens)),numAugs) 
            
            
            #If it is positive (as it is 2:1 inblanced positive)
            #Randomly decide to make the augmentation or not 
            #since we are effectively attempting to multiple the data by numAugs
            #and we want to even out the ration, the likely hood of 
            #doubling a positive class will be:
            #P=1 / (numAugs / 2)
            
            
            if c == "positive":
                posCnt+=1
                #skip this augmentation maybe
                if decision(1-probKeepPositive):
                    posSkipcnt+=1
                    continue
            if c == "neutral":
                #skip this augmentation maybe
                if decision(1-probKeepNeutral):
                    continue
            
                    
            #write that row as is:
            aug_data.append(row)
            
            #Keep track of the new data:
            cnt=0
            sucess_augs=0
            #make the augmentation, if failed retry, until we have no more chances
            for index in aug_indexs:
                #init for each augmentation
                fail=True
                tries=0
                cnt+=1
                while fail:
                    syns = getSyns(tokens[index])
                    if len(syns) < 1: #We failed to create a populated set
                        tries+=1
                        #pick a new index, don't try that one again:
                        try:
                            all_index.remove(index)
                        except IndexError:
                            #this index was already attempted and failed:
                            pass
                        choices = set(all_index) - set(aug_indexs)
                        try:
                            index = random.sample(choices,1)[0]
                        except ValueError: #we are out of choices
                            break
                        continue
                    
                    else: #we found a suitable augmentation:
                        newWord = random.sample(syns,1)[0]
                        if "_" in newWord:
                            #found synonm is two words:
                            replace = newWord.split("_")
                        else: #we make it into a list to make life easy 
                            replace = [newWord]
                        replacedWords = replacedWords + [list([tokens[index],newWord])]
                        
                        #add replaced words to our words in corpus:
                        for word in replace:
                            if word not in self.words_in_corpus:
                                self.words_in_corpus[word]=1
                            else:
                                self.words_in_corpus[word]+=1
                        ############################################
                        
                        tokens = tokens[:index] + replace + tokens[index+1:]
                        new_tweet = " ".join(tokens)
                        
                        #adjust ndx to know its augmented:
                        ndx = str(old_ndx) + "." + str(cnt)
                        
                        #write new row
                        new_row = [ndx,new_tweet,c]
                        aug_data.append(new_row) 
                        
                        #randomly decide to augment multiple words or 
                        #single word multiple times
                        if bool(random.getrandbits(1)):
                            #Do not replace (potentially) multi:
                            tokens = og_tokens
                        
                        #break out of loop for that augmentation
                        fail=False
                        
                        #do not reconsider this index:
                        all_index.remove(index)
                        
                    #kep track of how many sucessful augmentations we found
                    sucess_augs+=1
                
                ''' #Print failed augmentations
                if fail: #We couldnt find suitable synonym:
                    print("failed to find <",numAugs-sucess_augs," augmentations > in:")
                    print(" ".join(tokens),"\n")
                    break
                '''
                
        aug_df = pd.DataFrame(data=aug_data,columns=["ndx","Tweet","Sentiment"])
        
        if giveReplacements:
            return replacedWords, aug_df
        else:
            print("Skipped:",posSkipcnt/posCnt*100,"percent of positive augmentations")
            return -1, aug_df