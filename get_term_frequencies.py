#!/usr/bin/env python
# coding: utf-8


# Dictionary Exploration and Term Search with word2vec

# Charter School Identities Project
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: September 30, 2019
# Date last modified: September 30, 2019

# ## Initialize Python

# Import key packages
import gensim # for word embedding models
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
from collections import Counter # For counting terms across the corpus

# Import functions from other scripts
import sys; sys.path.insert(0, "../data_management/tools/") # To load functions from files in data_management/tools
from textlist_file import write_list, load_list # For saving and loading text lists to/from file
from df_tools import check_df, convert_df, load_filtered_df, replace_df_nulls # For displaying basic DF info, storing DFs for memory efficiency, and loading a filtered DF
from quickpickle import quickpickle_dump, quickpickle_load # For quickly loading & saving pickle files in Python
import count_dict # For counting word frequencies in corpus (to assess candidate words)
from count_dict import load_dict, Page, dict_precalc, dict_count, create_cols, count_words, collect_counts, count_master

# FOR CLEANING, TOKENIZING, AND STEMMING THE TEXT
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings


# Define file paths
home = '/vol_b/data/'
wem_path = home + 'wem4themes/data/wem_model_300dims.bin' # path to WEM model
charter_path = home + 'misc_data/charters_2015.pkl' # path to charter school data file
dict_path = home + 'text_analysis/dictionary_methods/dicts/' # path to dictionary files (may not be used here)

# For counting term frequencies, load text corpus:
df = load_filtered_df(charter_path, ["WEBTEXT", "NCESSCH"])
df['WEBTEXT']=df['WEBTEXT'].fillna('') # turn nan to empty iterable for future convenience


# ## Define helper functions

def dict_cohere(thisdict, wem_model):
    '''Computes the average cosine similarity score of terms within one dictionary with all other terms in that same dictionary,
    effectively measuring the coherence of the dictionary.
    ...question for development: does it make sense to compare the average cosine similarity score between all terms 
    in thisdict and the average cosine similarity among the total model vocabulary? (Could that be, by definition, 0?)
    
    NOTE: For an unknown reason, calling this function deletes terms from thisdict.
    
    Inputs: List of key terms, word2vec model.
    Output: Average cosine similarity score of each word with all other words in the list of key terms.'''
    
    # Initialize average distance variables:
    word_avg_dist = 0
    word_avg_dists = 0
    dict_avg_sim = 0
    all_avg_dists = 0
    model_avg_dists = 0
    
    # Compute average cosine similarity score of each word with other dict words:
    for word in thisdict:
        word_avg_dist = (wem_model.distances(word, other_words=thisdict).sum())/len(thisdict) # Total diffs of word with all other words, take average
        word_avg_dists += word_avg_dist # Add up each average distance, incrementally
    dict_avg_sim = 1 - word_avg_dists/len(thisdict) # Find average cosine similarity score by subtracting avg. distance from 1

    # For comparison, compute average cosine similarity score of each word with ALL other words in the model vocabulary:
    #for word in thisdict:
    #    all_avg_dist = (wem_model.distances(word).sum())/len(model.vocab) # Default is to compare each word with all words
    #    all_avg_dists += all_avg_dist
    #model_avg_dist = 1 - all_avg_dists/len(model.vocab) # Find average cosine similarity score by subtracting avg. distance from 1

    #print("Average cosine similarities by word for this dictionary:       \t" + str(dict_avg_dist))
    #print("Compare to avg. cosine similarities by dict words to ALL words:\t" + str(model_avg_dist))
    
    return dict_avg_sim

def focus_dict(thisdict, coredict, maxlen, wem_model):
    '''Focus thisdict by removing least similar word vectors until reaching maxlen.
    If any words from coredict get removed, compensate for fact that they will get added back in.
        
    Input: A list of terms, core terms not to remove, desired length, and word2vec model.
    Output: The input list focused down to desired length, and still containing all the core terms.'''

    core_count = 0 # Counts number of coredict terms that were removed
    extend_count = 0 # Counts number of terms removed to offset the coming boost of core terms (that were removed and will be added back in)

    while len(thisdict) > maxlen: # Narrow thisdict down to maxlen
        badvar = model.doesnt_match(thisdict) # Find least matching term
        thisdict.remove(badvar) # Remove that least focal term, to focus dict
        if badvar in coredict: # Keep track of number of core terms removed
            core_count += 1

    while extend_count < core_count: # Remove terms until length = maxlen - number of core terms removed (to offset those core terms that will be added back in later in this script)
        badvar = model.doesnt_match(thisdict) # Find least matching term
        thisdict.remove(badvar) # Remove that least focal term, to focus dict
        extend_count += 1 # Keep track of # non-core terms added
        if badvar in coredict: # Keep track of number of core terms removed
            core_count += 1
            
    for term in coredict: # Add back in any missing core terms
        if term not in thisdict and term in list(model.vocab):
            thisdict.append(term)
            
    thisdict = list(set(thisdict)) # Remove any duplicates
    
    if len(thisdict) != maxlen: # Quality check
        print("WARNING: Function produced a dictionary of length " + str(len(thisdict)) +               ", which is not the specified maximum dict length of " + str(maxlen))
    
    return thisdict 


# ## Load word embedding model

model = gensim.models.KeyedVectors.load_word2vec_format(wem_path, binary=True) # Load word2vec model


# ## Define dictionaries

# Core terms
inqseed = ['inquiry-based', 'problem-based', 'discovery-based', 'experiential', 'constructivist']

# By searching through the above candidate terms/phrases, expand from the seed terms into a conceptually tight list like this: 
inq30 = ['discovery-based', 'student-driven_exploration', 'exploration_and_experimentation', 'laboratory-based', 
         'problem-based', 'prbl', 'learn-by-doing', 
         'project-based', 'project-centered', 
         'experiential', 'experiential_approach', 'experientially',
         'inquiry-based', 'inquiry-driven', 'student-centered_inquiry-based', 'active_inquiry', 
         'constructivist', 'constructivism', 
         'hands-on', 'hand-on', 'hands-on_learning', 'hands-on_and_minds-on', 'hands-on_minds-on', 'hands-on/minds-on', 
         'socratic', 'socratic_method', 'socratic_dialogue',
         'child-centered', 'learner-centered', 'student-centered']

# Searching for additional terms similar to this list, you can expand even further!
inquiry_fin = [elem.strip('\n') for elem in load_list('data/inquiry.txt')] # Load completed dict of 500 terms
inquiry_fin = list(set(inquiry_fin)) # Remove duplicates

sorted(inquiry_fin) # Show long dictionary resulting from exploring (and hand-cleaning)

# Remove any terms from full dict NOT in current model:
for word in inquiry_fin:
    if word not in list(model.vocab):
        inquiry_fin.remove(word)
        print("Removed " + str(word) + " from core dictionary.")
# Repeat for quality:
for word in inquiry_fin:
    if word not in list(model.vocab):
        inquiry_fin.remove(word)
        print("Removed " + str(word) + " from core dictionary.")


# ## Check word frequencies and ranks

# ### Count frequency of candidate words
# 
# Required arguments for `count_master()` function:
# - df: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
# - dict_path: file path to folder containing dictionaries
# - dict_names: names of dictionaries on file (list or list of lists)
# - file_ext: file extension for dictionary files (probably .txt)   
# - local_dicts: list of local dictionaries formatted as last of lists of terms--or if singular, just a list of terms
# - local_names: names of local dictionaries (list or list of lists)

# Seed dictionary and similar terms:
candidate_sims = model.most_similar(inqseed, topn=500)
candidates_list = [pair[0] for pair in candidate_sims] + inqseed # Convert to list for frequency search below
countsdfs_IBL5sim = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [candidates_list], local_names = ["IBL_candidates"])
countsdfs_IBL5sim.to_csv("output/inquiry_seed_similar_counts.csv")

# 30-term IBL dictionary (core terms)
countsdfs_IBL30 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [inq30], local_names = ["IBL_candidates"])
countsdfs_IBL30.to_csv("output/inquiry_30_counts.csv")

# 30-term IBL dictionary and similar terms:
candidate_sims = model.most_similar(inq30, topn=500)
candidates_list = [pair[0] for pair in candidate_sims] + inq30 # Convert to list for frequency search below
countsdfs_IBL30sim = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [candidates_list], local_names = ["IBL_candidates"])
countsdfs_IBL30sim.to_csv("output/inquiry_30_similar_counts.csv")

# 500-term, unvalidated IBL dictionary:
countsdfs_IBL500 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [inquiry_fin], local_names = ["candidates"])
countsdfs_IBL500.to_csv("output/inquiry_500_counts.csv")