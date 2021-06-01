#!usr/bin/env python

"""
Automatic derivation of speech features from transcribed speech and its duration in sec (s):
Speech Rate, Articulation Rate, Character Count,  Article Count,
Average Characters Per Word, Sentence Length (Word Count), Syllable Count
& Average Syllable Per Word
"""

import argparse
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import re


def main(args: argparse.Namespace) -> None:
    
    #read file path and sheet name from excel to pandas dataframe
    def read_file(path, sheet):
        if type(path) == str and type(sheet) == str:
            return pd.read_excel(path, sheet_name=sheet)
        elif type(path) == str and sheet == None:
            return pd.read_excel(path)

    data = read_file(args.input, args.sheet)

    #Engineer Speech Features:
    #Function for removing common XML generated non-ascii character especially for Praat generated transcriptions
    def decode_string(string: str) -> str:
        encoded_string = string.encode("ascii", "ignore")
        decoded_string = encoded_string.decode()
        return re.sub(r'\s_x0019_',"'", decoded_string)

    #Cleaned Text for Analysis
    data['Cleaned_Text'] = data['Text'].apply(lambda x: decode_string(x))
    
    #Tokenize sentence into list of words
    data['Tokenized_Sent'] = data['Cleaned_Text'].apply(lambda x: (word_tokenize(x)))

    #Sentence Length
    data['Sent_Length'] = data['Cleaned_Text'].apply(lambda x: len(word_tokenize(x)))

    #Word Per Minute
    data['Speech_Rate'] = data['Cleaned_Text'].apply(lambda x: len(word_tokenize(x)))/data['Duration'] * 60

    #count syllables in word
    def syllable_count(word: str) -> int:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    #sum all syllables
    def syllable_count_list(words: list) -> int:
        count = []
        for w in words:
            count.append(syllable_count(w))
        return sum(count)

    #Sum of all syllables per senetnce
    data['Syllable_Length'] = data['Tokenized_Sent'].apply(lambda x: syllable_count_list(x))

    #Articulation rate (English): syllable per second
    data['Articulation_Rate'] = data['Syllable_Length']/data['Duration']

    #Syllables per word
    data['Syllable_Per_Word'] = data['Syllable_Length']/data['Sent_Length']

    #Concatenates all characters in a sentence
    data['Char'] = data['Cleaned_Text'].apply(lambda x: x.replace(" ", "").strip(","))

    #Total number of characters in a sentence
    data['Char_Length'] = data['Char'].apply(lambda x: len(x))
    
    #Characters per word
    data['Char_Per_Word'] = data['Char_Length']/data['Sent_Length']

    #Function for counting number of articles in a sentence
    def Count_Article(words: list) -> int:
        articles = ['a', 'the', 'an']
        count = 0
        for word in words:
            if word in articles:
                count += 1
        return count

    #Total number of articles per sentence
    data['Article_Count'] = data['Tokenized_Sent'].apply(lambda x : Count_Article(x))

    #Delete unnecessary columns
    data.drop('Char', axis=1, inplace=True)
    data.drop('Tokenized_Sent', axis=1, inplace=True)
    data.drop('Filename', axis=1, inplace=True)

    #Read output to excel file
    data.to_excel(args.output)
    print("Speech features have been derived. Check your output file path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Required: Input file path")
    parser.add_argument("--sheet", help="Optional: Input sheet name. E.g., inputfile --sheet sheetname outputfile")
    parser.add_argument("output", help="Required: Output file path")
    main(parser.parse_args())

"""Written by Gerald C. Imaezue"""
