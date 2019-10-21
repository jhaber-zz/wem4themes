#!/usr/bin/env python
# coding: utf-8


# Dictionary Exploration and Term Search with word2vec

# Charter School Identities Project
# Creator: Jaren Haber, PhD Candidate
# Institution: Department of Sociology, University of California, Berkeley
# Date created: September 30, 2019
# Date last modified: October 7, 2019

# ## Initialize Python

# Import key packages
import gensim # for word embedding models
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
from collections import Counter # For counting terms across the corpus
from scipy.spatial import distance # To use cosine distances for tSNE metric
import numpy as np # for working with vectors
import pandas as pd # To work with DFs

# Import functions from other scripts
import sys; sys.path.insert(0, "../data_management/tools/") # To load functions from files in data_management/tools
from textlist_file import write_list, load_list # For saving and loading text lists to/from file
from df_tools import check_df, convert_df, load_filtered_df, replace_df_nulls # For displaying basic DF info, storing DFs for memory efficiency, and loading a filtered DF
from quickpickle import quickpickle_dump, quickpickle_load # For quickly loading & saving pickle files in Python
#import count_dict # For counting word frequencies in corpus (to assess candidate words)
from count_dict import load_dict, Page, dict_precalc, dict_count, create_cols, count_words, collect_counts, count_master

# For cleaning, tokenizing, and stemming the text
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings


# Define file paths
home = '/vol_b/data/'
wem_path = home + 'wem4themes/data/wem_model_300dims.bin' # path to WEM model
charter_path = home + 'misc_data/charters_2015.pkl' # path to charter school data file
dict_path = home + 'text_analysis/dictionary_methods/dicts/' # path to dictionary files (may not be used here)

# For counting term frequencies, load text corpus:
print("Loading text corpus for term counting...")
df = load_filtered_df(charter_path, ["WEBTEXT", "NCESSCH"])
df['WEBTEXT']=df['WEBTEXT'].fillna('') # turn nan to empty iterable for future convenience


# ## Define dictionaries

riskseed = ["high-need", "low-income", "high-poverty", "at-risk", "high-risk"]

# Narrow dictionary for at-risk
risk20 = ["high-need", "high-needs", 
            "low-income", "lower-income", "high-poverty", "impoverished", "economically_challenged", 
            "underserved", "under-served", "disproportionately", 
            "at-risk", "high-risk", "under-resourced", "under-represented", "under-performing", 
            "inner-city", "inner_cities", 
            "marginalized", "disconnected", "disenfranchised"]

# Testing terms related to at-risk:
risktest = ['service', 'services', 'social_services', 'social_service', 'socialservices', 'diverse', 'diversity', 'cultural_diversity', 'culturally_diverse', 'culturallydiverse', 'family', 'district', 'dental', 'dental_services', 'dental_service', 'dentalservice', 'dentalservices', 'dental_care', 'dentalcare', 'medical', 'house', 'housing', 'cloth', 'clothe', 'clothes', 'clothing', 'ethnic', 'ethnic_identity', 'ethnicidentity', 'identity', 'health', 'employment', 'employment_services', 'employmentservices', 'economics', 'low-income', 'lowincome', 'low income', 'low_income', 'equality', 'equity', 'justice', 'disenfranchise', 'disenfranchised', 'disadvantage', 'disadvantaged', 'percentile', 'percentiles', 'quantile', 'quantiles', 'quintiles', 'quintile', 'quartile', 'quartiles', 'standard', 'support', 'supports', 'basic', 'basics', 'percent', 'urban', 'graduate', 'graduating', 'graduation', 'college', 'colleges', 'discipline', 'disciplinary', 'structural', 'structure', 'high-risk', 'at-risk', 'at_risk', 'at risk', 'represent', 'representation', 'under-represented', 'under_represented', 'under represented', 'need', 'high-needs', 'needs', 'high-need', 'high needs', 'great needs', 'under-resourced', 'underresourced', 'under_resourced', 'poverty', 'poor', 'impoverish', 'impoverished', 'under-achieving', 'underachieving', 'under_achieving', 'under achieving', 'African American', 'African', 'African-American', 'African_American', 'Hispanic', 'Hispanic-American', 'Latin', 'Latino', 'Latinx', 'Latin@', 'pregnant', 'impregnate', 'pregnancy', 'gifted', 'giftedprogram', 'gifted-program', 'gifted program', 'racism', 'racial', 'race', 'racist', 'racially', 'sexism', 'sexist', 'sex', 'generational', 'generation', 'cycle', 'cycles', 'chronic', 'chronically', 'food', 'nutrition', 'nutritious', 'nutritional', 'nutritive', 'nutrient', 'nutrients', 'prison', 'prisons', 'imprison', 'imprisoning', 'jail', 'jails', 'pipeline', 'pipe', 'incarcerate', 'incarceration', 'incarcerating', 'insecure', 'insecurity', 'assist', 'assistance', 'assisting', 'public_assistance', 'public-assistance', 'treatment', 'clinical', 'rehabilitate', 'rehabilitation', 'rehabilitated', 'rehabilitating', 'drug', 'drugs', 'drug_alcohol', 'drugs_and_alcohol', 'drugs and alcohol', 'case management', 'case_management', 'case-management', 'stable', 'stability', 'stabilize', 'stabilizing', 'counsel', 'counselling', 'counseling', 'counselor', 'counselors', 'behavior', 'behaviors', 'behavioral', 'continuum', 'continua', 'failure', 'failures', 'fail', 'failing', 'vulnerable', 'vulnerability', 'provide', 'provision', 'providing', 'provided', 'mental', 'mentality', 'substances', 'substance', 'patient', 'patients', 'abuse', 'abusive', 'abused', 'child abuse', 'abusing', 'dependent', 'dependency', 'dependencies', 'chemical', 'chemically', 'injury', 'illness', 'anxiety', 'anxious', 'anxieties', 'addiction', 'addictive', 'addict', 'addicts', 'punish', 'of color', 'of_color', 'people of color', 'of-color', 'people-of-color', 'people_of_color', 'out of school', 'out-of-school', 'out_of_school', 'truant', 'truancy', 'truism']

disc30 = ['absenteeism', 'authority', 'behavioral_expectations', 'crime', 'criminal', 'deter', 'deterrence', 'disciplinary', 'discipline', 'disobedience', 'drug-', 'related', 'drugs', 'expel', 'expellable', 'expulsion', 'illegal', 'inappropriate', 'misbehavior', 'no-excuses', 'penalize', 'penalty', 'perpetrator', 'punish', 'suspended', 'suspension', 'violate', 'violation', 'zero-tolerance', 'zero-tolerance_policy', 'zero_tolerance']

disc500 = ['absenteeism', 'authority', 'behavioral_expectations', 'crime', 'criminal', 'deter', 'deterrence', 'disciplinary', 'discipline', 'disobedience', 'drug-related', 'drugs', 'expel', 'expellable', 'expulsion', 'illegal', 'inappropriate', 'misbehavior', 'no-excuses', 'penalize', 'penalty', 'perpetrator', 'punish', 'suspended', 'suspension', 'violate', 'violation', 'zero-tolerance', 'zero-tolerance_policy', 'zero_tolerance', '120a.34', '121a.582', '121a.69', 'abandoning', 'absentee_voting', 'accusation', 'accusations', 'activity_has_a_dignifying', 'ad-hoc_committees', 'adequate_residence', 'adjudicated_delinquent', 'adverse_reaction', 'aggressor_target', 'aircraft_crash', 'alcohol', 'alimony_as_shown', 'allegation', 'allegations', 'alleged', 'altered_by_chemical', 'amendments_thereto', 'ammunition', 'anabolic_steroid', 'annexed', 'anthrax_spores', 'anti-retaliation', 'arbitration', 'arbitration_proceedings', 'arrested_convicted', 'arson', 'arson_or_criminal', 'articulable_and_significant', 'assault', 'assaulted', 'assaulting', 'assisted_or_represented', 'attached_to_the_duplicate', 'attempting_to_commit', 'attempts_to_kiss', 'authority_to_impose', 'aversive_techniques', 'barbiturate_marijuana', 'barbiturates', 'bare_skin', 'behavior/discipline', 'bestiality', 'bias_incident', 'bigoted', 'block_pores', 'bloodborne', 'bodily', 'box_and_whiskers', 'break/gross_motor', 'bullied', 'bully', 'bullying', 'butane', 'cardstock_white', 'carry_contraband', 'causing', 'certain_misdemeanors', 'chesting_a_person', 'claims_disputes', 'cleared_and_searched', 'closets_is_likely', 'coercing_forcing', 'colorism', 'commensurate_with_the_severity', 'complainant_or_witness', 'compulsive', 'concurred_in_by_the_principal', 'condones', 'conduct_a_headcount', 'confiscate', 'confiscation', 'consequences_for_the_perpetrator', 'considered_disapproved', 'considered_trespassing', 'construed_as_an_endorsement', 'contents_of_the_subpoena', 'contested_information', 'contractual_relationship', 'convicted', 'conviction', 'cool_pale', 'cough_wheeze', 'crayons_factory', 'cross-examined', 'culturally_biased', 'dagger', 'damage', 'danger', 'debt_incurred', 'deemed_truant', 'defamation_whether', 'deferred_pending', 'defibrillators', 'deliberate_attempts', 'demean_another', 'demeaning_jokes', 'denim_flannel', 'derogatory', 'design2fab_lab', 'designee_will_promptly', 'detriment', 'diately', 'disci', 'disciplinaryaction', 'discourteous', 'discriminatory', 'dishonor', 'disobey', 'dispense_possess', 'disrespectful', 'disrupt', 'disrupting', 'disruption', 'disruptive', 'disruptively', 'distributes_or_is_under_the_influence', 'district-affiliated', 'ditching_and_will_be_subject', 'dress_code', 'dwi_court', 'elderly_person', 'ellipsis_dash', 'ellipsis_points', 'ellipsis_to_indicate', 'embezzlement', 'endangering', 'endangers', 'endnotes', 'enhanced_with_adherence', 'entities_can_not_proceed', 'errors_in_capitalization', 'etaient_delicieuses', 'evaluate_the_motives', 'exam_is_paper-and-pencil', 'exhibiting_a_pattern', 'expended_on_its_behalf', 'export_csv', 'extortion_coercion', 'extreme_makeup', 'fact_grievable', 'failure_to_remedy', 'fainting_or_dizziness', 'false_accusation', 'false_charges', 'false_nails', 'fear_of_bodily', 'felony', 'femininity', 'fever-reducing_medicine', 'fighting', 'fighting_profanity', 'fighting_violent', 'firearm', 'fiscal_mismanagement', 'flavia', 'fluid-restrictive_diets', 'fmla_only_not_the_cfra', 'fondle', 'force_a_confrontation', 'formally_investigated', 'found_to_have_retaliated', 'fraud_or_fraudulent', 'french_verb', 'furnished_in_response', 'gallbladder', 'get_up-wind', 'glorifies', 'gloves/mittens', 'gross_misdemeanors', 'gun', 'gunfire', 'harass', 'harassed', 'harassing', 'harassment', 'harassment/bullying', 'harm', 'harmful', 'hazing', 'hibited', 'highly_addictive', 'homosexual_regardless', 'hostage', 'hostile', 'hostility', 'hotel_pans', 'identifies_with_moods', 'imminent', 'immobile', 'implantable', 'impose_consecutive', 'imposition_of_a_remedy', 'improvised_explosive', 'in-school_or_out-of-school', 'inadvertent_access', 'inattentive', 'incendiary_devices', 'incident/behavior', 'incident_of_violent/serious', 'incites', 'incorrigible', 'indecent_are_prohibited', 'inflammatory', 'inflict', 'inflicts', 'influence_of_marijuana', 'injure', 'injuring', 'instituted_against_a_repeated', 'interfere', 'interring', 'interrogate', 'interval_between_each_drill', 'intimidate', 'intimidates', 'intimidation', 'intoxicated', 'intoxication', 'intravenous', 'investigate', 'investigated_appropriately', 'investigation', 'invoked', 'involving_the_dispersion', 'itchy_eyes', 'judicial_decision', 'knife', 'knowingly_fabricate', 'lactose', 'laptop/netbook', 'large_stains', 'lawful', 'liable', 'liquids_with_caffeine', 'litigation_including', 'little_cigars', 'loaners', 'logos_embroidery', 'look-a-like', 'lowest_in_achievement', 'lunchroom_or_restroom', 'manslaughter_criminally', 'manufacture_distribute', 'manufacture_distribution', 'marijuana_other_than_concentrated', 'materials/equipment', 'mathworld', 'medical/dental_appointment', 'minor_on_both_sides', 'misconduct', 'mocking_taunting', 'moderate_risk', 'monies_owed', 'mood-altering', 'mouth_wash', 'municipal_ordinance', 'murder_manslaughter', 'narcotic', 'non-deliveries', 'non-fixed', 'non-forcible', 'non-renewal_by_an_employer', 'non-students_participating', 'non-work_related', 'notwithstanding_anything', 'oaths', 'obligation_to_be_truthful', 'obscene_abusive', 'obstruct', 'obstruct_the_investigation', 'obstruction', 'offending', 'offense', 'offensive', 'official_resigns', 'oily_stains', 'on-line_dictionary', 'orientation-actual', 'originated_in_cuba', 'outbreak_of_a_vaccine-preventable', 'package/letter', 'parchitecture_project', 'participates_in_a_proceeding', 'pattern_of_unwelcome', 'pay-as-you-go_method', 'pep-rally', 'permanent_exclusion', 'permits_condones', 'perpetrator_and_other_affected', 'pervasive_or_persistent', 'pervasively', 'pervision', 'pharmacy-labeled_container', 'physical_confinement', 'placement_of_video/audio', 'pled_no_contest', 'pocketknife', 'pointing_device', 'posed_a_significant', 'poses_an_immediate', 'possess_a_laser', 'possession', 'prefers_complexity', 'preoccupation_with_death', 'prerequisites_mth111', 'presumption_and_belief', 'prior_to_re-', 'ommending', 'privacy_defames', 'pro-rata', 'proceeding_for_the_purpose', 'profane_lewd', 'prohibited', 'prohibited_and_will_not_be_tolerated', 'prohibition_against_retaliation', 'projectiles', 'propellant', 'prosecute', 'prosecuted', 'prosecution', 'psychosocial_impairment', 'punctuation_comma', 'punishable', 'pursued_against_the_aggressor', 'putdowns', 'pyrotechnics', 'quell_a_disturbance', 'racist', 'readily_converted', 'reasonably_conclude', 'reasoning_and_the_relevance', 'reckless', 'recklessly', 'recklessness', 'reduce/stop', 'reflect_poorly', 'remains_dissatisfied', 'remediate_or_prevent', 'repeated_activation', 'repeated_intentional', 'reprimand', 'reprimanded', 'result_in_demerits', 'result_in_revocation', 'resulting_from_the_diversion', 'retained_in_anticipation', 'retaliating_against_an_employee', 'retaliation', 'retaliation_is_substantiated', 'retaliation_or_reprisal', 'revocation_of_privileges', 'revoked_and_disciplinary', 'ripped_torn', 'robo-calling', 'rock_throwing', 'rocky_planets', 'rule_adherence', 'saggy', 'sale/distribution', 'scaled_scores', 'scanned_for_viruses', 'school-support', 'scientific_procedures/experiments', 'sedatives', 'self-', 'ealing_transactions', 'self-injurious', 'self-motivation_leads', 'serious_or_persistent', 'settled_tribe', 'settlement_or_compromise', 'severity_and_frequency', 'sharpeners', 'similarity_and_scaling', 'simulate_the_effects', 'sleeveless_shirts/blouses', 'slight_adjustments', 'smoke_a_cigarette', 'smoking_in_a_pipe', 'spit_balls', 'stalking', 'starve', 'statutory', 'stigmatizing', 'strapless_tops', 'strength_vitality', 'subdivision_applies', 'subgrantee', 'subpoena_not_be_disclosed', 'substantial_emotional', 'substantiated_actssubstation', 'suffer_reprisals', 'suffers_from_clinical', 'suggestive', 'suspect', 'suspected', 'suspects', 'suspicion', 'swears', 'sweatpants_are_not_allowed', 'synthetic_leather', 'tagging', 'tampering_with_a_consumer', 'tardiness', 'tarnish', 'tattoos_temporary', 'tell/retell', 'temper_tantrum', 'testify_or_otherwise', 'textbooks_lockers', 'theft', 'theft/vandalism', 'thigh_buttocks', 'threat', 'threatened', 'threatening', 'threatens', 'three_musketeers', 'tire_deflation', 'tobacco', 'tolerate_hazing', 'tolerate_retaliation', 'toothache', 'touching', 'tracheostomy', 'trademark_infringement', 'transgression', 'truthful_and_that_retaliation', 'turkey_pastrami', 'unauthorized', 'unbecoming', 'unbidden_encounter', 'uncooperative', 'undesired', 'unidentified_rash', 'uniform', 'uniform/dress_code', 'unimproved_property', 'unlawful', 'unlawfully', 'unsafe_jewelry', 'unwanted_and_unwelcome', 'uphold_and_enforce', 'utilizable', 'vandalism', 'variation_of_these_traits', 'victim', 'vindicated', 'violating', 'violations', 'vulgar_profane', 'vulgar_violent', 'weapon', 'weapons', 'white-out', 'whoever_fails', 'whoever_knows', 'willful_infliction', 'willfully_damaging', 'willfully_defied', 'windhover_foundation', 'witchcraft', 'witness_or_witness', 'word_reference', 'worksites_is_prohibited', 'yelling_or_screaming', 'étude_sessions']

# Core DI terms
DIseed = []

direct_test = ['college-prep', 'direct-instruction', 'direct_instruction', 'drill', 'discipline', 'discipline-based', 'data-driven', 
               'facts', 'research-backed', 'memorization', 'rote', 'worksheets', 'highly-structured', 
               'no-excuses', 'core-knowledge', 'back-to-basics', 'knowledge-based', 'knowledge-rich', 
               'whole-group', 'teacher-centered', 'teacher-guided', 'didactic', 'standards-driven', 
               'literacy-based', 'literacy', 'seminar-style', 'curriculum-centered', 'mastery-learning', 'proven_methods', 'phonics']

# Core IBL terms
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
inquiry500 = [elem.strip('\n') for elem in load_list('data/inquiry.txt')] # Load completed dict of 500 terms
inquiry500 = list(set(inquiry500)) # Remove duplicates

'''
# Remove any terms from full dict NOT in current model:
for word in inquiry500:
    if word not in list(model.vocab):
        inquiry500.remove(word)
        print("Removed " + str(word) + " from core dictionary.")
# Repeat for quality:
for word in inquiry500:
    if word not in list(model.vocab):
        inquiry500.remove(word)
        print("Removed " + str(word) + " from core dictionary.")
'''

# ## Count word frequencies, save resulting DFs as CSV files

# ### Count frequency of candidate words
# 
# Required arguments for `count_master()` function:
# - df: DataFrame with text data, each of which is a list of full-text pages (not necessarily preprocessed)
# - dict_path: file path to folder containing dictionaries
# - dict_names: names of dictionaries on file (list or list of lists)
# - file_ext: file extension for dictionary files (probably .txt)   
# - local_dicts: list of local dictionaries formatted as list of lists of terms
# - local_names: names of local dictionaries (list or list of lists)

# Find similar terms for seed and 30-term dictionaries, convert to list for frequency search:
#candidate_sims = model.most_similar(inqseed, topn=5)
#similar_inqseed = [pair[0] for pair in candidate_sims] + inqseed
#candidate_sims = model.most_similar(inq30, topn=5)
#similar_inq30 = [pair[0] for pair in candidate_sims]

# Count term frequencies
#dicts_to_count = [inq30, similar_inqseed, similar_inq30, inquiry500]
#dict_names = ["IBL30", "IBLseed_similar", "IBL30_similar", "IBL500"]
#dicts_to_compare = [[], inqseed, inq30, []]

local_dicts = [risk20, risktest] #[inq30, similar_inqseed]
local_names =  ['atrisk20', 'atrisk_test'] #["IBL30", "IBLseed_similar"]
dicts_to_compare = [inqseed, inqseed, riskseed, riskseed]
dicts_from_file = ['inquiry20_new', 'inquiry50_new']

countsdfs = count_master(df, dict_path = dict_path, dict_names = dicts_from_file, file_ext = '.txt', 
                         local_dicts = local_dicts, local_names = local_names)

# Load dictionaries for easier computing and printing
local_dicts = [risk20, risktest] #[inq30, similar_inqseed]
local_names =  ['atrisk20', 'atrisk_test'] #["IBL30", "IBLseed_similar"]
dicts_to_compare = [inqseed, inqseed, riskseed, riskseed]
dicts_from_file = ['inquiry20_new', 'inquiry50_new']
file_dicts_number = len(dicts_from_file); local_dicts_number = len(local_names) # Makes comparisons faster

if file_dicts_number>0: # If there are dicts to be loaded from file...
    dicts_to_count = load_dict(dictpath = dict_path, dictnames = dicts_from_file, fileext = '.txt') # Load dictionaries from file
    dict_names = dicts_from_file
if file_dicts_number>0 and local_dicts_number>0: # If there are dicts on file AND local dicts...
    dicts_to_count += local_dicts # full list of dictionary names
    dict_names += local_names # full list of dictionaries
else: # If there are only local dicts...
    dicts_to_count = local_dicts
    dict_names = local_names

#file_dicts = load_dict(dictpath = dict_path, dictnames = dicts_from_file, fileext = ".txt")
#dicts_to_count = file_dicts + local_dicts
    
# Compute similarity of each candidate term the dict it's compared to, add to DF
if len(dicts_to_compare)>0:
    print("Computing similarities...")
    model = gensim.models.KeyedVectors.load_word2vec_format(wem_path, binary=True) # Load word2vec model 
    dicts_to_compare = [[word for word in dic if word in model.vocab] for dic in dicts_to_compare]
    dicts_to_count = [[word for word in dic if word in model.vocab] for dic in dicts_to_count]

for i, countdf in enumerate(countsdfs):
    if len(dicts_to_compare[i])>0:
        vectors = [model.get_vector(word) for word in dicts_to_compare[i]] # Get vectors for comparison dictionary
        average_vector = np.mean(vectors, axis=0) # Average comparison vectors
        similarities = [(1 - distance.cosine(average_vector, model.get_vector(word))) for word in dicts_to_count[i]] # Get similarities
        simdf = pd.DataFrame(similarities, index = dicts_to_count[i]) # Convert similarities to DF with terms as index
        print("Adding to DF...")
        #countdf["SIMILARITY"] = [(1 - distance.cosine(average_vector, model.get_vector(word))) for word in dicts_to_count[i]]
        countdf = pd.merge(countdf, simdf, how="left", left_on = "TERM", right_index = True) # Merge similarities with count df

    # Print outputs
    print("\nTERM COUNTS FOR " + str(dict_names[i].upper()) + " DICTIONARY:\n")
    print(countdf)

    # Save DF to disk
    countdf.to_csv('output/{}_counts.csv'.format(dict_names[i]))

'''
# 30-term IBL dictionary (core terms)
countsdfs_IBL30 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [inq30], local_names = ["IBL_candidates"])
countsdfs_IBL30[0].to_csv("output/inquiry_30_counts.csv")


countsdfs_IBL30sim = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [candidates_list], local_names = ["IBL_candidates"])
countsdfs_IBL30sim[0].to_csv("output/inquiry_30_similar_counts.csv")

# 500-term, unvalidated IBL dictionary:
countsdfs_IBL500 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [inquiry500], local_names = ["candidates"])
countsdfs_IBL500[0].to_csv("output/inquiry_500_counts.csv")
'''

'''
# test terms (possibly) related to "at-risk" concept
countsdfs_risktest = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [risk_test], local_names = ["risk_candidates"])
countsdfs_risktest[0].to_csv("output/risk_test_counts.csv")

# test terms (possibly) related to "direct-instruction" concept
countsdfs_directtest = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [direct_test], local_names = ["direct_candidates"])
countsdfs_directtest[0].to_csv("output/direct_test_counts.csv")

# 30-term, unvalidated disciplinary dictionary:
countsdfs_disc30 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [disc30], local_names = ["disc30"])
countsdfs_disc30[0].to_csv("output/disc30_counts.csv")

# 500-term, unvalidated disciplinary dictionary:
countsdfs_disc500 = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
                         local_dicts = [disc500], local_names = ["disc500"])
countsdfs_disc500[0].to_csv("output/disc500_counts.csv")
'''

# core DI dictionary and similar terms:
#candidate_sims = model.most_similar(DIseed, topn=500)
#candidates_list = [pair[0] for pair in candidate_sims] + DIseed # Convert to list for frequency search below
#countsdfs_DIseedsim = count_master(df, dict_path = dict_path, dict_names = [], file_ext = '.txt', 
#                         local_dicts = [candidates_list], local_names = ["IBL_candidates"])
#countsdfs_DIseedsim[0].to_csv("output/DI_seed_similar_counts.csv")