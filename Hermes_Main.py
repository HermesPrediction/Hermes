from pylab import *
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import os.path
import zipfile
import io
import email
import imaplib
import itertools
from collections import Counter
import hashlib
import pickle
import keras as k
from keras.layers import LSTM, Activation, Dropout, Dense, SimpleRNN, GRU, Bidirectional, Conv2D
from keras.models import load_model, Sequential
import os
import shutil
import signal


class Sequence(object):
    def __init__(self, email_address, seq_file):
        self.seq, self.seq_type = read_Seq_in(seq_file)
        for i in self.seq:
            self.seq[i] = ("".join(self.seq[i][0]))
        self.list_seq = self.seq
        self.str_seq = ''.join(self.seq.values())
        print("The DNA sequence is: {}".format(self.str_seq))
        self.email_address = email_address

    def __len__(self):
        self.len_seq = len(self.str_seq)
        return self.len_seq

    def Transcribe(self):
        self.RNA = RNA_Convert(self.list_seq)
        print("The RNA sequence is: {}".format(self.RNA))
        return self.RNA

    def Translate(self):
        self.protein = Translate_RNA(self.RNA, codon_table())
        print("Amino acids of the protein: {}".format("".join(str(self.protein))))
        print("Length of the protein: {}".format(len(self.protein)))
        return self.protein

    def Protein_Layer(self):
        protein_layer_predictor_list = \
            ['GORIV', 'PHD', 'NetSurfP', 'pS2', 'JPRED4', 'YASPIN', 'Spider3', 'RaptorX', 'SSpro', 'Porter4']
        self.protein_layer_predictor_list = protein_layer_predictor_list
        self.file_name, protein_file, self.protein_layer_list, self.saved_structure_layer_list, self.aggregator, self.saved_structure_list_flag, self.saved_aggregator_flag, self.active_predictors = Search_Processed_Files(
            self.protein)
        print("Protein Layer Predictors")
        if protein_file == None:
            self.active_predictors, self.protein_layer_list = Protein_Layer_Retrieval(self.email_address, self.protein,
                                                                                      self.file_name)
        else:
            for predictor, prediction in zip(self.protein_layer_predictor_list, self.protein_layer_list):
                print(predictor)
                print(prediction)
        print()
        if self.active_predictors[0] != ['GORIV_PHD_NetSurfP_pS2_JPRED4_YASPIN_Spider3_RaptorX_SSpro_Porter4']:
            print(self.active_predictors[0])
            print(self.active_predictors[1])
            print(
                "Unfortunately there is an issue with a protein layer server and an optimal result is unattainable at this momment, please try again later")
            exit()

    def Structure_Layer(self):
        structure_layer_predictor_list = \
            ['Weighted Vote', 'MLP', 'RNN', 'GRU', 'BiLSTM', 'CNN']
        self.structure_layer_predictor_list = structure_layer_predictor_list

        if self.saved_structure_list_flag == False:
            print("Structure Layer Predictors")
            inp = self.protein_layer_list
            structure_layer_list = [Structure_Layer_Weighted_Vote(self.active_predictors, inp),
                                    Structure_Layer_MLP(self.active_predictors, inp), Structure_Layer_RNN(self.active_predictors, inp),
                                    Structure_Layer_GRU(self.active_predictors, inp), Structure_Layer_BiLSTM(self.active_predictors, inp),
                                    Structure_Layer_CNN(self.active_predictors, inp)]
        else:
            structure_layer_list = self.saved_structure_layer_list

        self.aggregator_list = []
        for predictor, prediction in zip(structure_layer_predictor_list, structure_layer_list):
            self.aggregator_list.append(prediction)
            print(predictor)
            print(prediction)
            print()

    def Final_Hermes(self):
        self.protein_layer_list_filtered = list(filter(None, self.protein_layer_list))

        if self.saved_aggregator_flag == False:
            self.aggregator = Hermes_Aggregator(self.active_predictors, self.protein,
                                                [self.protein_layer_list_filtered[-2], self.protein_layer_list_filtered[-1], self.aggregator_list[0], self.aggregator_list[1],
                                                 self.aggregator_list[2], self.aggregator_list[3], self.aggregator_list[4], self.aggregator_list[5]])
        print('Final Hermes Solution')
        print("self.aggregator: {}".format(self.aggregator))

        with open('temp.txt', 'w') as infile_final:
            for prediction, predictor_name in zip(self.protein_layer_list, self.protein_layer_predictor_list):
                res = 0
                for residue in prediction:
                    if residue == "H":
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('acen')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')

                    elif residue == "E":
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('green')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')

                    elif residue == 'C':
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('stalk')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')
                    res += 1
            for prediction, predictor_name in zip(self.aggregator_list, self.structure_layer_predictor_list):
                res = 0
                for residue in prediction:
                    if residue == "H":
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('acen')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')

                    elif residue == "E":
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('green')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')

                    elif residue == 'C':
                        infile_final.write(predictor_name)
                        infile_final.write('     ')
                        infile_final.write(str(res))
                        infile_final.write('     ')
                        infile_final.write(str(res + 1))
                        infile_final.write('     ')
                        infile_final.write('stalk')
                        infile_final.write('     ')
                        infile_final.write(predictor_name + '\n')
                    res += 1
            res = 0
            for residue in self.aggregator:
                if residue == "H":
                    infile_final.write('Hermes')
                    infile_final.write('     ')
                    infile_final.write(str(res))
                    infile_final.write('     ')
                    infile_final.write(str(res + 1))
                    infile_final.write('     ')
                    infile_final.write('acen')
                    infile_final.write('     ')
                    infile_final.write('Hermes' + '\n')

                elif residue == "E":
                    infile_final.write('Hermes')
                    infile_final.write('     ')
                    infile_final.write(str(res))
                    infile_final.write('     ')
                    infile_final.write(str(res + 1))
                    infile_final.write('     ')
                    infile_final.write('green')
                    infile_final.write('     ')
                    infile_final.write('Hermes' + '\n')

                elif residue == 'C':
                    infile_final.write('Hermes')
                    infile_final.write('     ')
                    infile_final.write(str(res))
                    infile_final.write('     ')
                    infile_final.write(str(res + 1))
                    infile_final.write('     ')
                    infile_final.write('stalk')
                    infile_final.write('     ')
                    infile_final.write('Hermes' + '\n')
                res += 1
        self.file_name = Save_Prot_Predict(self.protein, 'temp')

        return self.file_name, self.protein, self.protein_layer_list, self.aggregator_list, self.aggregator


def read_Seq_in(filename):
    display = []
    with open(filename, "rU") as dna:
        for char in dna.read():
            char.upper()
            display.append("".join(char.strip()))

    input_Seq = []
    with open(filename, "rU") as f:
        for char in f.read():
            char.upper()
            input_Seq.append(char)

    def Residue_Check(sequence2check):
        seq_truth = ''
        for residue in sequence2check:

            if residue in ['A', 'T', 'G', 'C']:
                seq_truth += 'DNA'

            elif residue in ['K', 'S', 'H', 'R', 'D', 'V', 'P', 'M', 'W', 'F', 'Y', 'L', 'I', 'E', 'Q']:
                seq_truth += 'Protein'

        return seq_truth

    seq_truth = Residue_Check(input_Seq)

    if 'Protein' not in seq_truth:
        output_Seq = DNA_Template_2_Coding(str(input_Seq))
    else:
        output_Seq = input_Seq

    output_Seq = [value for value in output_Seq if value != '\n']

    Seq_Values = (list(x.upper() for x in output_Seq))
    Seq_Index = list(y for y in range(0, len(output_Seq)))

    Seq_Dict = {}

    for x, y in zip(Seq_Values, Seq_Index):
        Seq_Dict.setdefault(y, []).append(x)

    return Seq_Dict, seq_truth


def RNA_Convert(sequence1):
    seq1 = len(sequence1)

    if seq1 > 0:
        RNA = []
        for i in sequence1:
            if sequence1[i] == "T":
                RNA.append("A")
            elif sequence1[i] == "C":
                RNA.append("G")
            elif sequence1[i] == "G":
                RNA.append("C")
            elif sequence1[i] == "A":
                RNA.append("U")

        print()
        return ("".join(list(RNA)))


def DNA_Template_2_Coding(sequence):
    seq1 = len(sequence)
    if seq1 > 0:
        DNA1 = []

        for i in range(seq1):
            if sequence[i] == " ":
                DNA1.append("")
            if sequence[i] == "A" or sequence[i] == "a":
                DNA1.append("T")
            elif sequence[i] == "G" or sequence[i] == "g":
                DNA1.append("C")
            elif sequence[i] == "C" or sequence[i] == "c":
                DNA1.append("G")
            elif sequence[i] == "T" or sequence[i] == "t":
                DNA1.append("A")
        return ("".join(list(DNA1)))


def codon_table():
    genetic_encoding = {}
    single_letter = []

    with open("1Letter.txt", 'r') as one_letter:
        with open("triplet.txt", 'r') as triplet:

            for line in one_letter:
                single_letter.append(line.split())

            for i, line in enumerate(triplet):
                columns = line.split()
                genetic_encoding[columns[0]] = single_letter[i]

    return genetic_encoding


def Translate_RNA(RNA, gene_code):
    protein_seq = ''

    for i in range(len(RNA) // 3):
        start_codon = (i * 3)
        final_codon = start_codon + 3
        amino_acid = gene_code[RNA[start_codon:final_codon]]
        if "".join(amino_acid) != '!':
            protein_seq += "".join(amino_acid)
        else:
            break
    print()
    return protein_seq


def Protein_Layer_Retrieval(email_address, protein, file_name):
    name = str(file_name[:8])

    '''Call to GORIV'''
    URL = "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_gor4.pl"

    PARAMS = dict(title='Sequence',
                  notice=protein,
                  ali_width=70)

    r_GORIV = requests.post(url=URL, data=PARAMS,
                            headers={"Referer": "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_gor4.pl"})
    '''End of GORIV'''

    '''Call to PHD'''
    URL = "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_phd.pl"

    PARAMS = dict(title='Sequence',
                  notice=protein[:],
                  ali_width=70)

    r_PHD = requests.post(url=URL, data=PARAMS)
    '''End of PHD'''

    ''' Call to NetSurfP '''
    URL = "http://www.cbs.dtu.dk/cgi-bin/webface2.fcgi"

    payload = {
        'configfile': '/usr/opt/www/pub/CBS/services/NetSurfP-1.1/NetSurfP.cf',
        'pastefile': protein,
        'x': 'CACHE'}

    r_NetSurfP = requests.post(URL, data=payload)
    ''' End of NetSurfP'''

    ''' Call to pS2 '''
    URL1 = "http://ps2.life.nctu.edu.tw/ps2.php"
    PARAMS = dict(step='1', address=None, target=protein[:18], seq=protein, submit='SUBMIT', mode='auto')

    r_pS2 = requests.post(url=URL1, data=PARAMS, headers={"Referer": "http://sparks-lab.org/server/SPIDER3/index.php"})
    ''' End of pS2 '''

    ''' Call to JPRED4 '''
    URL = "http://www.compbio.dundee.ac.uk/jpred/cgi-bin/jpred_form"
    payload = {
        'seq': protein,
        'input': 'seq',
        'pdb': 'on',
        'email': email_address,
        'queryName': protein[:18],
        '.submit': 'continue'}
    r_JPRED4 = requests.post(URL, data=payload)
    ''' End of JPRED4 '''

    ''' Call to YASPIN '''
    URL = "http://www.ibi.vu.nl/programs/sympredwww/"
    sequence = '>blahblah\n' + protein
    Payload = {
        'seq': sequence,
        'mbjob[description]': 'SYMPRED Job',
        'database': 'nr',
        'pred2': '-prof',
        'pred5': '-yaspin',
        'conmethod': 'D',
        'conweight': 'N',
        'sympred': 'Do prediction'}
    r_YASPIN = requests.post(url=URL, data=Payload)
    ''' End of YASPIN'''

    ''' Call to SPIDER3'''
    URL = "http://sparks-lab.org/server/SPIDER3/index.php"

    PARAMS = dict(REPLY=None, TARGET='Sequence', SEQUENCE=protein, METHOD='SPIDER3')

    r_Spider3 = requests.post(url=URL, data=PARAMS,
                              headers={"Referer": "http://sparks-lab.org/server/SPIDER3/index.php"})
    '''End of Spider3'''

    ''' Call to RaptorX'''
    URL = "http://raptorx2.uchicago.edu/StructurePropertyPred/predict/"

    PARAMS = dict(jobname=protein[:18], email=email_address, sequences=protein, fileseq=None, useProfile='true',
                  predict_sub='Submit')

    r_RaptorX = requests.post(url=URL, data=PARAMS,
                              headers={"Referer": "http://raptorx2.uchicago.edu/StructurePropertyPred/predict/"})
    ''' End of RaptorX'''

    ''' Start of SSpro '''
    URL = "http://scratch.proteomics.ics.uci.edu/cgi-bin/new_server/sql_predict.cgi"

    PARAMS = dict(email=email_address, query_name=name,
                  amino_acids=protein, ss='on')

    r_SSpro = requests.post(url=URL, data=PARAMS, headers={"Referer": "http://scratch.proteomics.ics.uci.edu/"})

    with open("SSpro_Sec", "w") as infile:
        for char in r_SSpro.text:
            infile.write(char)
    with open("SSpro_Sec", "r") as infile:
        for char in range(len(r_SSpro.text)):
            if r_SSpro.text[char] == 'E' and r_SSpro.text[char + 1] == 'R' and r_SSpro.text[char + 2] == 'R' and \
                            r_SSpro.text[char + 5] == ':' and r_SSpro.text[char + 7] == 'Y' and r_SSpro.text[
                        char + 12] == 'J' and \
                            r_SSpro.text[char + 20] == 'N' and r_SSpro.text[char + 29] == 'S' and r_SSpro.text[
                        char + 31] == 'B':
                time.sleep(120)
                r_SSpro = requests.post(url=URL, data=PARAMS,
                                        headers={"Referer": "http://scratch.proteomics.ics.uci.edu/"})
                break
    ''' End of SSpro '''

    '''Call to Porter4'''
    URL = "http://distillf.ucd.ie/~distill/cgi-bin/distill/predict"

    PARAMS = dict(email_address=email_address, input_name=protein[:18], porter='secondary',
                  input_text=protein)

    r_Porter4 = requests.post(url=URL, data=PARAMS, headers={"Referer": "http://distill.ucd.ie/porter/home.html"})
    '''End of Porter4'''

    def Protein_Layer_Seq_Retrieval(email_address, protein, name, r_GORIV, r_PHD, r_NetSurfP, r_pS2, r_JPRED4, r_YASPIN,
                                    r_Spider3, r_RaptorX):
        class Timeout:
            def __init__(self, seconds):
                self.seconds = seconds

            def Predictor_Drop_Out_Counter(count):
                updated_count = count - 1
                return updated_count

            def Raise_Error(self, signum, frame):
                raise TimeoutError()

            def __enter__(self):
                signal.signal(signal.SIGALRM, self.Raise_Error)
                signal.alarm(self.seconds)

            def __exit__(self, type, val, traceback):
                signal.alarm(0)

        prot_name = 'temp'
        rolling_time = 1000

        current_count = 10
        ''' Start of GOR IV '''
        start1 = time.time()
        Str_GORIV_Predict = ''
        try:
            with Timeout(rolling_time):
                soup = BeautifulSoup(r_GORIV.text, "lxml")

                for a in soup.find_all('a', href=True, text=True):
                    link_text = a['href']

                URL = 'https://npsa-prabi.ibcp.fr/' + link_text

                r_GORIV = requests.get(url=URL)

                with open("GORIV_Sec", "w") as infile:
                    for char in r_GORIV.text:
                        infile.write(char)

                NPS_Predict = []
                with open("GORIV_Sec", 'r') as infile:
                    text = infile.readlines()
                    for j in (text):
                        NPS_Predict.append(j[0])

                print('GORIV')
                Str_GORIV_Predict = DSSP_8_to_3("".join((NPS_Predict[4:])))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_GORIV_Predict = ''
            pass
        end1 = time.time()
        elapsed_time1 = int(end1 - start1)
        ''' End of GOR IV '''

        '''Start of PHD'''
        start2 = time.time()
        rolling_time = (rolling_time - elapsed_time1) + 1000
        Str_PHD_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                PHD_Flag = False
                PHD_Flag1 = False

                with open("PHD_Sec", "w") as infile:
                    for char in r_PHD.text:
                        infile.write(char)

                with open("PHD_Sec", "a") as infile:
                    for char in range(len(r_PHD.text)):
                        if r_PHD.text[char] == 'E' and r_PHD.text[char + 1] == 'R' \
                                and r_PHD.text[char + 2] == 'R' and r_PHD.text[char + 3] == 'O' \
                                and r_PHD.text[char + 4] == 'R' and r_PHD.text[char + 7] == 'c':

                            # Retrying a post with first residue concatendated from GORIV
                            URL = "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_phd.pl"

                            PARAMS = dict(title='Sequence',
                                          notice=protein[1:],
                                          ali_width=70)

                            r_PHD = requests.post(url=URL, data=PARAMS)
                            with open("PHD_Sec", "w") as infile:
                                for char in r_PHD.text:
                                    infile.write(char)
                            PHD_Flag = True

                            for char in range(len(r_PHD.text)):
                                if r_PHD.text[char] == 'E' and r_PHD.text[char + 1] == 'R' \
                                        and r_PHD.text[char + 2] == 'R' and r_PHD.text[char + 3] == 'O' \
                                        and r_PHD.text[char + 4] == 'R' and r_PHD.text[char + 7] == 'c':

                                    # Retrying a post with first two residues concatendated from GORIV
                                    URL = "https://npsa-prabi.ibcp.fr/cgi-bin/secpred_phd.pl"

                                    PARAMS = dict(title='Sequence',
                                                  notice=protein[2:],
                                                  ali_width=70)

                                    r_PHD = requests.post(url=URL, data=PARAMS)
                                    with open("PHD_Sec", "w") as infile:
                                        for char in r_PHD.text:
                                            infile.write(char)
                                    PHD_Flag1 = True
                            break

                link_text = []
                soup = BeautifulSoup(r_PHD.text, "lxml")

                for a in soup.find_all('a', href=True, text=True):
                    link_text.append(a['href'])

                URL = 'https://npsa-prabi.ibcp.fr' + link_text[10]

                r_PHD = requests.get(url=URL)

                with open("PHD_Sec", "w") as infile:
                    for char in r_PHD.text:
                        infile.write(char)

                NPS_Predict = []
                with open("PHD_Sec", 'r') as infile:
                    text = infile.readlines()
                    for j in (text):
                        for char in range(len(j)):
                            if j[char] == 'P' and j[char + 1] == 'H' and j[char + 2] == 'D':
                                NPS_Predict.append(j)

                data = []
                flag = False
                with open('PHD_Sec', 'r') as f:
                    for line in f:
                        if line.startswith('         PHD |'):
                            flag = True
                        if flag:
                            data.append(line[14:-2])
                        if line.strip().endswith('|'):
                            flag = False

                data = ''.join(data)

                Str_PHD_Predict = ''
                for char in data:
                    if char == 'H':
                        Str_PHD_Predict += 'H'

                    elif char == 'E':
                        Str_PHD_Predict += 'E'

                    elif char == ' ':
                        Str_PHD_Predict += 'C'

                print('PHD')
                if PHD_Flag1 == True:
                    Str_PHD_Predict = Str_GORIV_Predict[:2] + Str_PHD_Predict
                    Str_PHD_Predict = DSSP_8_to_3(Str_PHD_Predict)
                elif PHD_Flag == True:
                    Str_PHD_Predict = Str_GORIV_Predict[:1] + Str_PHD_Predict
                    Str_PHD_Predict = DSSP_8_to_3(Str_PHD_Predict)
                else:
                    Str_PHD_Predict = DSSP_8_to_3(Str_PHD_Predict)
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_PHD_Predict = ''
            pass
        end2 = time.time()
        elapsed_time2 = int(end2 - start2)
        ''' End of PHD'''

        ''' Start of NetSurfP '''
        start3 = time.time()
        rolling_time = (rolling_time - elapsed_time2) + 1000
        Str_NetSurfP_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                link_text = []

                soup = BeautifulSoup(r_NetSurfP.text, "lxml")
                for a in soup.find_all('a', href=True, text=True):
                    link_text.append(a['href'])

                URL2 = link_text[0]

                r_NetSurfP = requests.get(url=URL2)
                NetSurfP_len = len(r_NetSurfP.text)
                with open("NetSurfP_Sec", "w") as infile:
                    for char in r_NetSurfP.text:
                        infile.write(char)

                with open("NetSurfP_Sec", "r") as infile:
                    for char in range(NetSurfP_len):
                        if r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                and r_NetSurfP.text[char + 25] == 'u':
                            while r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                    and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                    and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                    and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                    and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                    and r_NetSurfP.text[char + 25] == 'u':
                                time.sleep(60)
                                r_NetSurfP = requests.get(url=URL2)
                                NetSurfP_len = len(r_NetSurfP.text)
                            else:
                                break

                    for char in range(NetSurfP_len):
                        if r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                and r_NetSurfP.text[char + 25] == 'u':
                            while r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                    and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                    and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                    and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                    and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                    and r_NetSurfP.text[char + 25] == 'u':
                                time.sleep(30)
                                r_NetSurfP = requests.get(url=URL2)
                                NetSurfP_len = len(r_NetSurfP.text)
                            else:
                                break

                    for char in range(NetSurfP_len):
                        if r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                and r_NetSurfP.text[char + 25] == 'u':
                            while r_NetSurfP.text[char] == 'T' and r_NetSurfP.text[char + 1] == 'h' \
                                    and r_NetSurfP.text[char + 2] == 'i' and r_NetSurfP.text[char + 3] == 's' \
                                    and r_NetSurfP.text[char + 5] == 'p' and r_NetSurfP.text[char + 8] == 'e' \
                                    and r_NetSurfP.text[char + 11] == 'h' and r_NetSurfP.text[char + 15] == 'd' \
                                    and r_NetSurfP.text[char + 17] == 'r' and r_NetSurfP.text[char + 21] == 'a' \
                                    and r_NetSurfP.text[char + 25] == 'u':
                                time.sleep(10)
                                r_NetSurfP = requests.get(url=URL2)
                            else:
                                break

                r_NetSurfP = requests.get(url=URL2)
                with open("NetSurfP_Sec", "w") as infile:
                    for char in r_NetSurfP.text:
                        infile.write(char)

                flag_NetSurfP = False
                NetSurfP_Predict = []
                with open("NetSurfP_Sec", 'r') as infile:
                    for line in infile.readlines():
                        if line.startswith('# Column 10: Probability for Coil'):
                            flag_NetSurfP = True
                        if flag_NetSurfP:
                            NetSurfP_Predict.append(line)
                        if line.strip().endswith('<font face="ARIAL,HELVETICA">'):
                            flag_NetSurfP = False

                Coil = []
                Helix = []
                Beta = []
                Total = []
                y = []
                List_NetSurfP = []
                element_count = (len(NetSurfP_Predict[1]))

                for element in range(len(NetSurfP_Predict) - 2):
                    element += 1
                    Coil2 = ((int(NetSurfP_Predict[element][-4:element_count - 1])), 'C')
                    Coil.append(Coil2)
                    Helix2 = ((int(NetSurfP_Predict[element][-12:element_count - 9])), 'B')
                    Helix.append(Helix2)
                    Beta2 = ((int(NetSurfP_Predict[element][-20:element_count - 16])), 'H')
                    Beta.append(Beta2)
                    Total.append((Coil2, Helix2, Beta2))

                for res in (range(len(Total))):
                    y.append(max(Total[res], key=lambda x: x[0]))
                    List_NetSurfP.append(y[res][1])

                print('NetSurfP')
                Str_NetSurfP_Predict = DSSP_8_to_3(''.join(List_NetSurfP))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_NetSurfP_Predict = ''
            pass
        end3 = time.time()
        elapsed_time3 = int(end3 - start3)
        ''' End of NetSurfP'''

        ''' Start of JPRED4 '''
        start4 = time.time()
        rolling_time = (rolling_time - elapsed_time3) + 1000
        Str_JPRED4_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                JPRED4_link_text = []
                soup = BeautifulSoup(r_JPRED4.text, "lxml")
                for a in soup.find_all('a', href=True, text=True):
                    JPRED4_link_text.append(a['href'])
                URL = 'http://www.compbio.dundee.ac.uk/jpred4/results/' + JPRED4_link_text[9][54:64] + '/' + \
                      JPRED4_link_text[9][
                      54:64] + '.simple.html'

                r_JPRED4 = requests.get(url=URL)
                with open("JPRED4_Sec", "w") as infile:
                    for char in r_JPRED4.text:
                        infile.write(char)

                with open("JPRED4_Sec", "r") as infile:
                    for char in range(len(r_JPRED4.text)):
                        if r_JPRED4.text[char] == 'q' and r_JPRED4.text[char + 1] == 'u' \
                                and r_JPRED4.text[char + 2] == 'e' and r_JPRED4.text[char + 3] == 's' \
                                and r_JPRED4.text[char + 4] == 't' and r_JPRED4.text[char + 8] == 'U' \
                                and r_JPRED4.text[char + 13] == 's' and r_JPRED4.text[char + 17] == 't':
                            while r_JPRED4.text[char] == 'q' and r_JPRED4.text[char + 1] == 'u' \
                                    and r_JPRED4.text[char + 2] == 'e' and r_JPRED4.text[char + 3] == 's' \
                                    and r_JPRED4.text[char + 4] == 't' and r_JPRED4.text[char + 8] == 'U' \
                                    and r_JPRED4.text[char + 13] == 's' and r_JPRED4.text[char + 17] == 't':
                                time.sleep(1)
                                r_JPRED4 = requests.get(url=URL)
                            else:
                                break

                with open("JPRED4_Sec", "w") as infile:
                    for char in r_JPRED4.text:
                        infile.write(char)

                flag_JPRED4 = False
                JPRED4_Predict = []
                JPRED4_Predict_final = ''
                with open("JPRED4_Sec", 'r') as infile:
                    for line in infile.readlines():
                        if line.startswith('</head><body><pre><code>'):
                            flag_JPRED4 = True
                        if flag_JPRED4:
                            JPRED4_Predict.append(line)
                        if line.strip().endswith('</code></pre>'):
                            flag_JPRED4 = False

                for char in (JPRED4_Predict[1]):
                    if char == 'H' or char == '-' or char == 'E':
                        JPRED4_Predict_final += char

                print("JPRED4")
                Str_JPRED4_Predict = DSSP_8_to_3(JPRED4_Predict_final)
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_JPRED4_Predict = ''
            pass
        end4 = time.time()
        elapsed_time4 = int(end4 - start4)
        ''' End of JPRED4'''

        ''' Start of pS2 '''
        start5 = time.time()
        rolling_time = (rolling_time - elapsed_time4) + 1000
        Str_pS2_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                URL1 = "http://ps2.life.nctu.edu.tw/ps2.php"

                PARAMS = dict(step='1', address=None, target=protein[:18], seq=protein, submit='SUBMIT', mode='auto')

                r = requests.post(url=URL1, data=PARAMS,
                                  headers={"Referer": "http://sparks-lab.org/server/SPIDER3/index.php"})

                href_link = []

                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.find_all('a', href=True, text=True):
                    href_link.append(a.get('href'))

                URL = 'http://ps2.life.nctu.edu.tw/display2.' + href_link[4][4:24]

                r = requests.get(url=URL)

                count1 = 0
                mover = 0
                with open("pS2_Sec", "w") as infile:
                    for char in r.text:
                        infile.write(char)

                with open("pS2_Sec", "r") as infile:
                    for char in range(len(r.text)):
                        if r.text[char] == 'c' and r.text[char + 1] == 'l' \
                                and r.text[char + 7] == 't' and r.text[char + 42] == 'f' \
                                and r.text[char + 54] == 't' and r.text[char + 39] == '0':
                            while r.text[char] == 'c' and r.text[char + 1] == 'l' \
                                    and r.text[char + 7] == 't' and r.text[char + 42] == 'f' \
                                    and r.text[char + 54] == 't' and r.text[char + 39] == '0':
                                time.sleep(1)
                                mover += 6
                                count1 += 1
                                r = requests.post(url=URL1, data=PARAMS,
                                                  headers={"Referer": "http://sparks-lab.org/server/SPIDER3/index.php"})
                                soup = BeautifulSoup(r.text, "lxml")

                                for a in soup.find_all('a', href=True, text=True):
                                    href_link.append(a.get('href'))
                                URL = 'http://ps2.life.nctu.edu.tw/display2.' + href_link[mover + 4][4:24]
                                r = requests.get(url=URL)
                            else:
                                break

                count = 0
                with open("pS2_Sec", "w") as infile:
                    for char in r.text:
                        infile.write(char)

                with open("pS2_Sec", "a") as infile:
                    for char in range(len(r.text)):
                        if r.text[char] == 'd' and r.text[char + 1] == 'i' \
                                and r.text[char + 5] == 'l' and r.text[char + 10] == '/' \
                                and r.text[char + 16] == 'r' and r.text[char + 20] == '<' \
                                and r.text[char + 23] == 'd':
                            while r.text[char] == 'd' and r.text[char + 1] == 'i' \
                                    and r.text[char + 5] == 'l' and r.text[char + 10] == '/' \
                                    and r.text[char + 16] == 'r' and r.text[char + 20] == '<' \
                                    and r.text[char + 23] == 'd':
                                time.sleep(1)
                                count += 1
                                r = requests.get(url=URL)

                            else:
                                break

                r = requests.get(url=URL)

                with open("pS2_Sec", "w") as infile:
                    for char in r.text:
                        infile.write(char)

                flag_pS2 = False
                pS2_Parse = []
                with open("pS2_Sec", "r") as infile:
                    for line in infile.readlines():
                        if line.startswith(
                                "<table border='0' align='center' width='950' style='word-break:break-all'><tr valign='top'><td class='table_header' width='120'><div align='left'"):
                            flag_pS2 = True
                        if flag_pS2:
                            pS2_Parse.append(line)
                        if line.strip().endswith('</textarea></td></tr><tr'):
                            flag_pS2 = False

                length = len(protein) + 1
                pS2_Parse = ''.join(pS2_Parse)
                final = []
                for char in range(len(pS2_Parse)):
                    if pS2_Parse[char] == 'd' and pS2_Parse[char + 1] == 'i' \
                            and pS2_Parse[char + 5] == 'l' and pS2_Parse[char + 8] == '>':
                        final.append(pS2_Parse[char + 9:(char + length + 8)])

                print('pS2')
                Str_pS2_Predict = DSSP_8_to_3(''.join(final[1]))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_pS2_Predict = ''
            pass
        end5 = time.time()
        elapsed_time5 = int(end5 - start5)
        ''' End of pS2 '''

        ''' Start of YASPIN'''
        start6 = time.time()
        rolling_time = (rolling_time - elapsed_time5) + 1000
        Str_YASPIN_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):

                URL2 = str(r_YASPIN.url)
                r_YASPIN = requests.get(url=URL2)

                with open("YASPIN_Sec", "w") as infile:
                    for char in r_YASPIN.text:
                        infile.write(char)

                with open("YASPIN_Sec", "r") as infile:
                    for char in range(len(r_YASPIN.text)):
                        if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                            while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                time.sleep(15)
                                r_YASPIN = requests.get(url=URL2)
                            else:
                                break

                r_YASPIN = requests.get(url=URL2)

                with open("YASPIN_Sec", "r") as infile:
                    for char in range(len(r_YASPIN.text)):
                        if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                            while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                time.sleep(10)
                                r_YASPIN = requests.get(url=URL2)
                            else:
                                break

                r_YASPIN = requests.get(url=URL2)

                with open("YASPIN_Sec", "r") as infile:
                    for char in range(len(r_YASPIN.text)):
                        if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':

                            while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                time.sleep(10)
                                r_YASPIN = requests.get(url=URL2)
                            else:
                                break

                URL3 = URL2 + 'result.yaspin'
                r_YASPIN = requests.get(url=URL3)

                with open("YASPIN_Sec", 'w') as infile:
                    for char in r_YASPIN.text:
                        infile.write(char)

                YASPIN_data = []
                with open('YASPIN_Sec', 'r') as outfile:
                    for line in outfile.readlines():
                        if line.startswith(" Pred: "):
                            YASPIN_data.append(line[7:-1])

                Str_YASPIN_Predict = ''.join(YASPIN_data)

                while Str_YASPIN_Predict == None or Str_YASPIN_Predict == '' or len(Str_YASPIN_Predict) != len(protein):

                    with open('dummy_protein_call_file.txt', 'w') as prot_file:
                        prot_file.write('>Q03096 | EST3_YEAST (Telomere replication protein EST3)')
                        prot_file.write('\n')
                        for char in protein:
                            prot_file.write(char)

                    URL = "http://www.ibi.vu.nl/programs/yaspinwww/"

                    files = {'seq_file': open('dummy_protein_call_file.txt', 'r')}

                    payload = {
                        'seq': 'MVVGGTEAQRNSWPSQISLQYRSGSSWAHTCGGTLIRQNWVMTAAHCVDRELTFRVVVGEHNLNQNNGTEQYVGVQKIVVHPYWNTDDVAAGYDIALLRLAQSVTLNSYVQLGVLPRAGTILANNSPCYITGWGLTRTNGQLAQTLQQAYLPTVDYAICSSSSYWGSTVKNSMVCAGGDGVRSGCQGDSGGPLHCLVNGQYAVHGVTSFVSRLGCNVTRKPTVFTRVSAYISWINNVIAS',
                        'mbjob[description]': 'YASPIN Job',
                        'smethod': 'nr',
                        'nnmethod': 'dssp',
                        'yaspin_align': 'YASPIN prediction'}

                    r_YASPIN = requests.post(URL, files=files, data=payload)
                    URL2 = str(r_YASPIN.url)
                    r_YASPIN = requests.get(url=URL2)

                    with open("YASPIN_Sec", "w") as infile:
                        for char in r_YASPIN.text:
                            infile.write(char)

                    with open("YASPIN_Sec", "r") as infile:
                        for char in range(len(r_YASPIN.text)):
                            if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                        and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                        and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                        and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                        and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                        and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                    time.sleep(15)
                                    r_YASPIN = requests.get(url=URL2)
                                else:
                                    break

                    r_YASPIN = requests.get(url=URL2)
                    with open("YASPIN_Sec", "r") as infile:
                        for char in range(len(r_YASPIN.text)):
                            if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                        and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                        and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                        and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                        and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                        and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                    time.sleep(10)
                                    r_YASPIN = requests.get(url=URL2)
                                else:
                                    break

                    r_YASPIN = requests.get(url=URL2)

                    with open("YASPIN_Sec", "r") as infile:
                        for char in range(len(r_YASPIN.text)):
                            if r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                    and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                    and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                    and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                    and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                    and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                while r_YASPIN.text[char] == 'R' and r_YASPIN.text[char + 1] == 'e' \
                                        and r_YASPIN.text[char + 2] == 's' and r_YASPIN.text[char + 3] == 'u' \
                                        and r_YASPIN.text[char + 4] == 'l' and r_YASPIN.text[char + 8] == 'w' \
                                        and r_YASPIN.text[char + 11] == 'l' and r_YASPIN.text[char + 13] == 'a' \
                                        and r_YASPIN.text[char + 16] == 'e' and r_YASPIN.text[char + 20] == 'h' \
                                        and r_YASPIN.text[char + 25] == 'a' and r_YASPIN.text[char + 30] == 'a':
                                    time.sleep(10)
                                    r_YASPIN = requests.get(url=URL2)
                                else:
                                    break

                    r_YASPIN = requests.get(url=URL2)

                    with open("YASPIN_Sec", 'w') as infile:
                        for char in r_YASPIN.text:
                            infile.write(char)

                    YASPIN_data = []
                    YASPIN_flag = False
                    with open('YASPIN_Sec', 'r') as outfile:
                        for line in outfile.readlines():
                            if line.startswith(
                                    "<tr><td><font face='Courier New, Courier, mono' size=2><b>Prediction:</b></td><td><font face='Courier New, Courier, mono' size=2>"):
                                YASPIN_flag = True
                            if YASPIN_flag:
                                for char in line:
                                    if char == '-' or char == 'H' or char == 'E':
                                        YASPIN_data.append(char)
                            if line.strip().endswith('</font></td></tr>'):
                                YASPIN_flag = False
                    Str_YASPIN_Predict = ''.join(YASPIN_data)

                print('YASPIN')
                Str_YASPIN_Predict = DSSP_8_to_3(''.join(Str_YASPIN_Predict))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            pass
        end6 = time.time()
        elapsed_time6 = int(end6 - start6)
        ''' End of YASPIN'''

        ''' Start of SPIDER3'''
        start7 = time.time()
        rolling_time = (rolling_time - elapsed_time6) + 1000
        Str_Spider3_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                href_link2 = []

                soup = BeautifulSoup(r_Spider3.text, "lxml")

                for a in soup.find_all('a', href=True, text=True):
                    href_link2.append(a.get('href'))

                URL = 'http://sparks-lab.org/server/SPIDER3/' + href_link2[1]

                r_Spider3 = requests.get(url=URL)
                href_link2 = []
                with open("Spider3_Sec", "w") as infile:
                    for char in r_Spider3.text:
                        infile.write(char)

                with open("Spider3_Sec", "a") as infile:
                    for char in range(len(r_Spider3.text)):
                        if r_Spider3.text[char] == 's' and r_Spider3.text[char + 1] == 'u' \
                                and r_Spider3.text[char + 2] == 'b' and r_Spider3.text[char + 3] == 'm' \
                                and r_Spider3.text[char + 4] == 'i' and r_Spider3.text[char + 5] == 's' \
                                and r_Spider3.text[char + 11] == 'i' and r_Spider3.text[char + 12] == 's' \
                                and r_Spider3.text[char + 14] == 'D' and r_Spider3.text[char + 16] == 'p':
                            soup2 = BeautifulSoup(r_Spider3.text, "lxml")
                            for a in soup2.find_all('a', href=True, text=True):
                                href_link2.append(a.get('href'))
                            URL = 'http://sparks-lab.org/server/SPIDER3/' + href_link2[0][24:28] + href_link2[0][28:]
                            r_Spider3 = requests.get(url=URL)

                    for char in range(len(r_Spider3.text)):
                        if r_Spider3.text[char] == 'P' and r_Spider3.text[char + 1] == 'l' \
                                and r_Spider3.text[char + 2] == 'e' and r_Spider3.text[char + 3] == 'a' \
                                and r_Spider3.text[char + 4] == 's' and r_Spider3.text[char + 5] == 'e' \
                                and r_Spider3.text[char + 7] == 'w':
                            while r_Spider3.text[char] == 'P' and r_Spider3.text[char + 1] == 'l' \
                                    and r_Spider3.text[char + 2] == 'e' and r_Spider3.text[char + 3] == 'a' \
                                    and r_Spider3.text[char + 4] == 's' and r_Spider3.text[char + 5] == 'e' \
                                    and r_Spider3.text[char + 7] == 'w':
                                time.sleep(1)
                                r_Spider3 = requests.get(url=URL)
                            else:
                                break

                with open("Spider3_Sec", "w") as infile:
                    for char in r_Spider3.text:
                        infile.write(char)

                Spider3_Predict = []
                with open("Spider3_Sec", 'r') as infile:
                    text = infile.readlines()
                    for j in (text):
                        if j[:2] == 'SS':
                            for char in j:
                                if char == 'H' or char == '-' or char == 'E':
                                    Spider3_Predict.append(char)

                print("Spider3")
                Str_Spider3_Predict = DSSP_8_to_3("".join(Spider3_Predict))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_Spider3_Predict = ''
            pass
        end7 = time.time()
        elapsed_time7 = int(end7 - start7)
        ''' End of SPIDER3'''

        ''' Start of RAPTORX '''
        start8 = time.time()
        rolling_time = (rolling_time - elapsed_time7) + 1000
        Str_RaptorX_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                href_link2 = []

                soup = BeautifulSoup(r_RaptorX.text, "lxml")

                for a in soup.find_all('a', href=True, text=True):
                    href_link2.append(a.get('href'))

                URL = href_link2[7]
                r_RaptorX = requests.get(url=URL)

                count = 0
                with open("RaptorX_Sec", "w") as infile:
                    for char in r_RaptorX.text:
                        infile.write(char)

                with open("RaptorX_Sec", "a") as infile:
                    for char in range(len(r_RaptorX.text)):
                        if r_RaptorX.text[char] == 'p' and r_RaptorX.text[char + 1] == 'r' \
                                and r_RaptorX.text[char + 2] == 'o' and r_RaptorX.text[char + 3] == 'c' \
                                and r_RaptorX.text[char + 4] == 'e' and r_RaptorX.text[char + 5] == 's' \
                                and r_RaptorX.text[char + 11] == 'i' and r_RaptorX.text[char + 12] == 'v' \
                                and r_RaptorX.text[char - 3] == 'n' and r_RaptorX.text[char - 5] == 'e':
                            while r_RaptorX.text[char] == 'p' and r_RaptorX.text[char + 1] == 'r' \
                                    and r_RaptorX.text[char + 2] == 'o' and r_RaptorX.text[char + 3] == 'c' \
                                    and r_RaptorX.text[char + 4] == 'e' and r_RaptorX.text[char + 5] == 's' \
                                    and r_RaptorX.text[char + 11] == 'i' and r_RaptorX.text[char + 12] == 'v' \
                                    and r_RaptorX.text[char - 3] == 'n' and r_RaptorX.text[char - 5] == 'e':
                                time.sleep(1)
                                count += 1
                                r_RaptorX = requests.get(url=URL)

                            else:
                                break

                download_link = []

                with open("RaptorX_Sec", "w") as infile:
                    for char in range(len(r_RaptorX.text)):
                        infile.write(r_RaptorX.text[char])

                    for char in range(len(r_RaptorX.text)):
                        if r_RaptorX.text[char] == '/' and r_RaptorX.text[char + 1] == 'S' and \
                                        r_RaptorX.text[char + 2] == 't' and r_RaptorX.text[char + 3] == 'r' \
                                and r_RaptorX.text[char + 46] == 't' and r_RaptorX.text[char + 47] == 'a':
                            download_link.append(r_RaptorX.text[char:char + 48])

                URL2 = 'http://raptorx2.uchicago.edu/' + download_link[0]
                r_RaptorX = requests.get(URL2, stream=True)

                z = zipfile.ZipFile(io.BytesIO(r_RaptorX.content))

                Dir1 = str(os.getcwd()) + '/'

                z.extractall(Dir1)

                RaptorX_Predict = []

                file_name = Dir1 + URL[-6:] + "/" + URL[-6:] + ".ss3_simp.txt"

                with open(file_name, 'r') as infile:
                    text = infile.readlines()
                    for j in text:
                        RaptorX_Predict.append(j)

                folder_name = Dir1 + URL[-6:]
                try:
                    shutil.rmtree(folder_name)
                except:
                    pass

                print('RaptorX')
                Str_RaptorX_Predict = DSSP_8_to_3("".join((RaptorX_Predict[2])))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_RaptorX_Predict = ''
            pass
        end8 = time.time()
        elapsed_time8 = int(end8 - start8)
        ''' End of RAPTORX '''

        ''' Start of SSpro '''
        start9 = time.time()
        rolling_time = (rolling_time - elapsed_time8) + 1000
        Str_SSpro_Predict = ''
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                user = email_address
                pwd = ("Congratulations")

                m = imaplib.IMAP4_SSL("imap.gmail.com")
                m.login(user, pwd)

                m.select("INBOX")

                subject_SSpro = '(FROM "baldig@ics.uci.edu" SUBJECT "Protein Structure Predictions for ' + name[
                                                                                                           :8] + '")'

                resp, data = m.search(None, subject_SSpro)
                data_test = True

                while data_test == True:

                    time.sleep(1)
                    m = imaplib.IMAP4_SSL("imap.gmail.com")
                    m.login(user, pwd)
                    m.select("INBOX")
                    resp, data = m.search(None, subject_SSpro)
                    if data == [b'']:
                        data_test = True
                        continue

                    else:
                        break

                for num in data[0].split():
                    typ, data = m.fetch(num, '(RFC822)')
                    SSpro_msg = email.message_from_string(data[0][1].decode('utf-8'))

                SSpro_msg = str(SSpro_msg)
                SSpro = SSpro_msg[
                        SSpro_msg.find("(3 Class):") + 10:SSpro_msg.find("If you requested a SVM contact map")]
                SSpro = SSpro.split()

                print('SSpro')
                Str_SSpro_Predict = DSSP_8_to_3("".join(SSpro))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_SSpro_Predict = ''
            pass
        end9 = time.time()
        elapsed_time9 = int(end9 - start9)
        ''' End of SSpro '''

        ''' Start of Porter4 '''
        Str_Porter4_Predict = ''
        rolling_time = (rolling_time - elapsed_time9) + 10000
        if rolling_time <= 0:
            rolling_time = 0
        try:
            with Timeout(rolling_time):
                user = email_address
                pwd = ("Congratulations")

                m = imaplib.IMAP4_SSL("imap.gmail.com")
                m.login(user, pwd)

                m.select("INBOX")

                subject_Porter4 = '(FROM "gianluca.pollastri@ucd.ie" SUBJECT "Porter response to ' + protein[:18] + '")'

                resp, data = m.search(None, subject_Porter4)

                data_test = True

                while data_test == True:
                    time.sleep(1)
                    m = imaplib.IMAP4_SSL("imap.gmail.com")
                    m.login(user, pwd)
                    m.select("INBOX")
                    resp, data = m.search(None, subject_Porter4)
                    if data == [b'']:
                        data_test = True
                        continue

                    else:
                        break

                for num in data[0].split():
                    typ, data = m.fetch(num, '(RFC822)')
                    msg = email.message_from_string(data[0][1].decode('utf-8'))

                msg = str(msg)

                print('Porter4')
                mySubString = msg[msg.find("Prediction:") + 12:msg.find("Predictions")]
                mySubString = mySubString.split()

                Str_Porter4_Predict = DSSP_8_to_3("".join(mySubString[1::2]))
        except:
            current_count = Timeout.Predictor_Drop_Out_Counter(current_count)
            Str_Porter4_Predict = ''
            pass
        ''' End of Porter4 '''

        predictions = {
            Str_GORIV_Predict: 'GORIV',
            Str_PHD_Predict: 'PHD',
            Str_NetSurfP_Predict: 'NetSurfP',
            Str_pS2_Predict: 'pS2',
            Str_JPRED4_Predict: 'JPRED4',
            Str_YASPIN_Predict: 'YASPIN',
            Str_Spider3_Predict: 'Spider3',
            Str_RaptorX_Predict: 'RaptorX',
            Str_SSpro_Predict: 'SSpro',
            Str_Porter4_Predict: 'Porter4'}

        with open(prot_name + '.txt', 'w') as mot:
            mot.write("")

        with open(prot_name + '.txt', 'a') as mot:
            for prediction, predictor_name in predictions.items():
                res = 0
                for residue in prediction:
                    if residue == "H":
                        mot.write(predictor_name)
                        mot.write('     ')
                        mot.write(str(res))
                        mot.write('     ')
                        mot.write(str(res + 1))
                        mot.write('     ')
                        mot.write('acen')
                        mot.write('     ')
                        mot.write(predictor_name + '\n')

                    elif residue == "E":
                        mot.write(predictor_name)
                        mot.write('     ')
                        mot.write(str(res))
                        mot.write('     ')
                        mot.write(str(res + 1))
                        mot.write('     ')
                        mot.write('green')
                        mot.write('     ')
                        mot.write(predictor_name + '\n')

                    elif residue == 'C':
                        mot.write(predictor_name)
                        mot.write('     ')
                        mot.write(str(res))
                        mot.write('     ')
                        mot.write(str(res + 1))
                        mot.write('     ')
                        mot.write('stalk')
                        mot.write('     ')
                        mot.write(predictor_name + '\n')
                    res += 1

                HTML_file_name = predictor_name + "_Sec"
                if os.path.isfile(HTML_file_name):
                    os.remove(HTML_file_name)
                else:
                    pass

                if element == 'Porter4':
                    if os.path.isfile("dummy_protein_call_file.txt"):
                        os.remove("dummy_protein_call_file.txt")
                    else:
                        pass

        Save_Prot_Predict(protein, prot_name)

        list_of_protein_layer = [Str_GORIV_Predict, Str_PHD_Predict, Str_NetSurfP_Predict, Str_pS2_Predict,
                                 Str_JPRED4_Predict, Str_YASPIN_Predict, Str_Spider3_Predict, Str_RaptorX_Predict,
                                 Str_SSpro_Predict, Str_Porter4_Predict]

        joined_names = ''
        for prediction, predictor in zip(list_of_protein_layer,
                                         ['GORIV', 'PHD', 'NetSurfP', 'pS2', 'JPRED4', 'YASPIN', 'Spider3', 'RaptorX',
                                          'SSpro', 'Porter4']):
            if prediction != '':
                joined_names += predictor + '_'

        return [joined_names[:-1], current_count, protein], [Str_GORIV_Predict, Str_PHD_Predict, Str_NetSurfP_Predict,
                                                             Str_pS2_Predict, Str_JPRED4_Predict, Str_YASPIN_Predict,
                                                             Str_Spider3_Predict, Str_RaptorX_Predict,
                                                             Str_SSpro_Predict, Str_Porter4_Predict]

    return Protein_Layer_Seq_Retrieval(email_address, protein, name, r_GORIV, r_PHD, r_NetSurfP, r_pS2, r_JPRED4,
                                       r_YASPIN, r_Spider3, r_RaptorX)


def Structure_Layer_Weighted_Vote(active_preds, protein_layer_predictions):
    predictor_combination = active_preds[0].split('_')

    GORIV = protein_layer_predictions[0]
    PHD = protein_layer_predictions[1]
    NetSurfP = protein_layer_predictions[2]
    pS2 = protein_layer_predictions[3]
    JPRED4 = protein_layer_predictions[4]
    YASPIN = protein_layer_predictions[5]
    Spider3 = protein_layer_predictions[6]
    RaptorX = protein_layer_predictions[7]
    SSpro = protein_layer_predictions[8]
    Porter4 = protein_layer_predictions[9]

    predictor_ratio_dict = {}

    if "GORIV" in predictor_combination:
        predictor_ratio_dict["GORIV"] = [GORIV]

    if 'PHD' in predictor_combination:
        predictor_ratio_dict["PHD"] = [PHD]

    if 'NetSurfP' in predictor_combination:
        predictor_ratio_dict["NetSurfP"] = [NetSurfP, NetSurfP]

    if 'pS2' in predictor_combination:
        predictor_ratio_dict["pS2"] = [pS2]

    if 'JPRED4' in predictor_combination:
        predictor_ratio_dict["JPRED4"] = [JPRED4]

    if 'YASPIN' in predictor_combination:
        predictor_ratio_dict["YASPIN"] = [YASPIN]

    if 'Spider3' in predictor_combination:
        predictor_ratio_dict["Spider3"] = [Spider3, Spider3]

    if 'RaptorX' in predictor_combination:
        predictor_ratio_dict["RaptorX"] = [RaptorX, RaptorX, RaptorX]

    if 'SSpro' in predictor_combination:
        predictor_ratio_dict["SSpro"] = [SSpro, SSpro, SSpro, SSpro]

    if "Porter4" in predictor_combination:
        predictor_ratio_dict["Porter4"] = [Porter4, Porter4, Porter4, Porter4]

    predictor_ratios = []
    for predictor in predictor_combination:
        predictor_ratios.append(predictor_ratio_dict[predictor])

    total_list = []
    for char1 in zip(predictor_ratios):
        for unit in char1:
            for element in unit:
                total_list.append(element)

    print(total_list)
    print(total_list[0])
    print(len(total_list[0]))

    aligned = []
    for i in range(len(total_list[0])):
        temp_aligned = []
        for res in total_list:
            temp_aligned.append(res[i])
        aligned.append(temp_aligned)

    consensus = []
    for res in aligned:
        x = Counter(res)
        y = x.most_common(1)
        consensus.append(y[0][0])

    Weighted_Vote_Consensus = (''.join(consensus))

    return Weighted_Vote_Consensus


def Structure_Layer_MLP(active_preds, protein_layer_predictions):
    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    clf_file = open(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_MLP_" + active_preds[0] + '.pickle', 'rb')
    clf = pickle.load(clf_file)
    clf_file.close()

    np.set_printoptions(threshold=np.inf)

    protein_array = np.array([], dtype=np.int)
    GORIV_num = []
    PHD_num = []
    NetSurfP_num = []
    pS2_num = []
    JPRED4_num = []
    YASPIN_num = []
    Spider3_num = []
    RaptorX_num = []
    SSpro_num = []
    Porter4_num = []

    for prediction, array in zip(protein_layer_predictions,
                                 [GORIV_num, PHD_num, NetSurfP_num, pS2_num, JPRED4_num, YASPIN_num, Spider3_num,
                                  RaptorX_num, SSpro_num, Porter4_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    for char1, char2, char3, char4, char5, char6, char7, char8, char9, char10 in itertools.zip_longest(GORIV_num,
                                                                                                       PHD_num,
                                                                                                       NetSurfP_num,
                                                                                                       pS2_num,
                                                                                                       JPRED4_num,
                                                                                                       YASPIN_num,
                                                                                                       Spider3_num,
                                                                                                       RaptorX_num,
                                                                                                       SSpro_num,
                                                                                                       Porter4_num):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])
        protein_array = np.append(protein_array, [char9])
        protein_array = np.append(protein_array, [char10])

    protein_array = protein_array[protein_array != np.array(None)]

    a = protein_array.reshape(len(protein_layer_predictions[-1]), active_preds[1])
    prediction = clf.predict(a)
    prob_results = clf.predict_proba(a)

    MLP_Consensus = ''
    for element, unit in zip(prediction, prob_results):
        if element == 1:
            MLP_Consensus += 'H'
        elif element == 2:
            MLP_Consensus += 'E'
        elif element == 0:
            MLP_Consensus += 'C'

    return MLP_Consensus


def Structure_Layer_RNN(active_preds, protein_layer_predictions):
    k.backend.tensorflow_backend.clear_session()

    np.set_printoptions(threshold=np.inf)

    protein_array = np.array([], dtype=np.int)
    GORIV_num = []
    PHD_num = []
    NetSurfP_num = []
    pS2_num = []
    JPRED4_num = []
    YASPIN_num = []
    Spider3_num = []
    RaptorX_num = []
    SSpro_num = []
    Porter4_num = []

    for prediction, array in zip(protein_layer_predictions,
                                 [GORIV_num, PHD_num, NetSurfP_num, pS2_num, JPRED4_num, YASPIN_num, Spider3_num,
                                  RaptorX_num, SSpro_num, Porter4_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    for char1, char2, char3, char4, char5, char6, char7, char8, char9, char10 in itertools.zip_longest(GORIV_num,
                                                                                                       PHD_num,
                                                                                                       NetSurfP_num,
                                                                                                       pS2_num,
                                                                                                       JPRED4_num,
                                                                                                       YASPIN_num,
                                                                                                       Spider3_num,
                                                                                                       RaptorX_num,
                                                                                                       SSpro_num,
                                                                                                       Porter4_num):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])
        protein_array = np.append(protein_array, [char9])
        protein_array = np.append(protein_array, [char10])

    protein_array = protein_array[protein_array != np.array(None)]
    a = protein_array.reshape(len(active_preds[2]), 1, active_preds[1])

    model = k.Sequential()
    model.add(SimpleRNN(active_preds[1], return_sequences=True, trainable=True, input_shape=(1, active_preds[1])))
    model.add(SimpleRNN(128, trainable=True))
    model.add(Dropout(0.225))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = k.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.01)

    model.compile(optimizer=opt, loss='categorical_crossentropy')

    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    # edit 105 to new file name
    model = load_model(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_RNN_" + active_preds[0] + ".tfl")

    pred = model.predict(a)

    prediction = [np.argmax(y, axis=None, out=None) for y in pred]
    RNN_Consensus = ''

    for element, unit in zip(prediction, pred):
        if element == 1:
            RNN_Consensus += 'H'
        elif element == 2:
            RNN_Consensus += 'E'
        else:
            RNN_Consensus += 'C'

    return RNN_Consensus


def Structure_Layer_GRU(active_preds, protein_layer_predictions):
    k.backend.tensorflow_backend.clear_session()

    np.set_printoptions(threshold=np.inf)

    protein_array = np.array([], dtype=np.int)
    GORIV_num = []
    PHD_num = []
    NetSurfP_num = []
    pS2_num = []
    JPRED4_num = []
    YASPIN_num = []
    Spider3_num = []
    RaptorX_num = []
    SSpro_num = []
    Porter4_num = []

    for prediction, array in zip(protein_layer_predictions,
                                 [GORIV_num, PHD_num, NetSurfP_num, pS2_num, JPRED4_num, YASPIN_num, Spider3_num,
                                  RaptorX_num, SSpro_num, Porter4_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    for char1, char2, char3, char4, char5, char6, char7, char8, char9, char10 in itertools.zip_longest(GORIV_num,
                                                                                                       PHD_num,
                                                                                                       NetSurfP_num,
                                                                                                       pS2_num,
                                                                                                       JPRED4_num,
                                                                                                       YASPIN_num,
                                                                                                       Spider3_num,
                                                                                                       RaptorX_num,
                                                                                                       SSpro_num,
                                                                                                       Porter4_num):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])
        protein_array = np.append(protein_array, [char9])
        protein_array = np.append(protein_array, [char10])

    protein_array = protein_array[protein_array != np.array(None)]
    a = protein_array.reshape(len(active_preds[2]), 1, active_preds[1])

    model = k.Sequential()
    model.add(GRU(active_preds[1], input_shape=(1, active_preds[1]), return_sequences=True))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(256))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy')

    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    # edit 105 to new file name
    model = load_model(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_GRU_" + active_preds[0] + ".tfl")

    pred = model.predict(a)

    prediction = [np.argmax(y, axis=None, out=None) for y in pred]
    GRU_Consensus = ''

    for element, unit in zip(prediction, pred):
        if element == 1:
            GRU_Consensus += 'H'
        elif element == 2:
            GRU_Consensus += 'E'
        else:
            GRU_Consensus += 'C'

    return GRU_Consensus


def Structure_Layer_BiLSTM(active_preds, protein_layer_predictions):
    k.backend.tensorflow_backend.clear_session()
    np.set_printoptions(threshold=np.inf)

    protein_array = np.array([], dtype=np.int)
    GORIV_num, New_GORIV = [], []
    PHD_num, New_PHD = [], []
    NetSurfP_num, New_NetSurfP = [], []
    pS2_num, New_pS2 = [], []
    JPRED4_num, New_JPRED4 = [], []
    YASPIN_num, New_YASPIN = [], []
    Spider3_num, New_Spider3 = [], []
    RaptorX_num, New_RaptorX = [], []
    SSpro_num, New_SSpro = [], []
    Porter4_num, New_Porter4 = [], []

    time_steps = 50
    batch = 25

    for prediction, array in zip(protein_layer_predictions,
                                 [GORIV_num, PHD_num, NetSurfP_num, pS2_num, JPRED4_num, YASPIN_num, Spider3_num,
                                  RaptorX_num, SSpro_num, Porter4_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    if 'GORIV' in active_preds[0]:
        New_GORIV = np.pad(GORIV_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'PHD' in active_preds[0]:
        New_PHD = np.pad(PHD_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'NetSurfP' in active_preds[0]:
        New_NetSurfP = np.pad(NetSurfP_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'pS2' in active_preds[0]:
        New_pS2 = np.pad(pS2_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'JPRED4' in active_preds[0]:
        New_JPRED4 = np.pad(JPRED4_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'YASPIN' in active_preds[0]:
        New_YASPIN = np.pad(YASPIN_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'Spider3' in active_preds[0]:
        New_Spider3 = np.pad(Spider3_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'RaptorX' in active_preds[0]:
        New_RaptorX = np.pad(RaptorX_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'SSpro' in active_preds[0]:
        New_SSpro = np.pad(SSpro_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'Porter4' in active_preds[0]:
        New_Porter4 = np.pad(Porter4_num, (0, 10000 - len(active_preds[2])), 'constant')
    predict_Batch_count = ((((10000 * active_preds[1]) // time_steps) // batch) // active_preds[1])

    for char1, char2, char3, char4, char5, char6, char7, char8, char9, char10 in itertools.zip_longest(New_GORIV,
                                                                                                       New_PHD, New_pS2,
                                                                                                       New_NetSurfP,
                                                                                                       New_JPRED4,
                                                                                                       New_YASPIN,
                                                                                                       New_Spider3,
                                                                                                       New_RaptorX,
                                                                                                       New_SSpro,
                                                                                                       New_Porter4):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])
        protein_array = np.append(protein_array, [char9])
        protein_array = np.append(protein_array, [char10])

    protein_array = protein_array[protein_array != np.array(None)]
    a = protein_array.reshape(predict_Batch_count, batch, time_steps, active_preds[1])

    model = k.Sequential()

    model.add(Bidirectional(LSTM(active_preds[1], return_sequences=True, stateful=True),
                            input_shape=(time_steps, active_preds[1]),
                            batch_input_shape=(batch, time_steps, active_preds[1])))
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=True)))
    model.add(Dropout(0.15))
    model.add(k.layers.TimeDistributed(Dense(active_preds[1])))
    model.add(Dense(active_preds[1]))
    model.add(Dropout(0.1))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.add(k.layers.ActivityRegularization(l1=0.0, l2=0.0))

    opt = k.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.01)

    model.compile(optimizer=opt, loss='categorical_crossentropy')

    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    # edit 105 to new file name
    model = load_model(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_BiLSTM_" + active_preds[0] + ".tfl")

    final_prob = []
    BiLSTM_Consensus = ''
    for element in range(len(a)):
        prob1 = model.predict(a[element])
        Prediction_Proba = model.predict_classes(a[element])
        for array in Prediction_Proba:
            for unit in range(len(array)):

                if array[unit] == 1:
                    BiLSTM_Consensus += 'H'
                elif array[unit] == 2:
                    BiLSTM_Consensus += 'E'
                else:
                    BiLSTM_Consensus += 'C'
        final_prob.append(prob1)

    BiLSTM_Consensus = BiLSTM_Consensus[:len(protein_layer_predictions[-1])]

    return BiLSTM_Consensus


def Structure_Layer_CNN(active_preds, protein_layer_predictions):
    k.backend.tensorflow_backend.clear_session()

    np.set_printoptions(threshold=np.inf)

    protein_array = np.array([], dtype=np.int)
    GORIV_num, New_GORIV = [], []
    PHD_num, New_PHD = [], []
    NetSurfP_num, New_NetSurfP = [], []
    pS2_num, New_pS2 = [], []
    JPRED4_num, New_JPRED4 = [], []
    YASPIN_num, New_YASPIN = [], []
    Spider3_num, New_Spider3 = [], []
    RaptorX_num, New_RaptorX = [], []
    SSpro_num, New_SSpro = [], []
    Porter4_num, New_Porter4 = [], []

    batch = 20

    for prediction, array in zip(protein_layer_predictions,
                                 [GORIV_num, PHD_num, NetSurfP_num, pS2_num, JPRED4_num, YASPIN_num, Spider3_num,
                                  RaptorX_num, SSpro_num, Porter4_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    if 'GORIV' in active_preds[0]:
        New_GORIV = np.pad(GORIV_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'PHD' in active_preds[0]:
        New_PHD = np.pad(PHD_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'NetSurfP' in active_preds[0]:
        New_NetSurfP = np.pad(NetSurfP_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'pS2' in active_preds[0]:
        New_pS2 = np.pad(pS2_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'JPRED4' in active_preds[0]:
        New_JPRED4 = np.pad(JPRED4_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'YASPIN' in active_preds[0]:
        New_YASPIN = np.pad(YASPIN_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'Spider3' in active_preds[0]:
        New_Spider3 = np.pad(Spider3_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'RaptorX' in active_preds[0]:
        New_RaptorX = np.pad(RaptorX_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'SSpro' in active_preds[0]:
        New_SSpro = np.pad(SSpro_num, (0, 10000 - len(active_preds[2])), 'constant')
    if 'Porter4' in active_preds[0]:
        New_Porter4 = np.pad(Porter4_num, (0, 10000 - len(active_preds[2])), 'constant')
    predict_Batch_count = (((10000 * active_preds[1]) // batch) // active_preds[1])

    for char1, char2, char3, char4, char5, char6, char7, char8, char9, char10 in itertools.zip_longest(New_GORIV,
                                                                                                       New_PHD,
                                                                                                       New_NetSurfP,
                                                                                                       New_pS2,
                                                                                                       New_JPRED4,
                                                                                                       New_YASPIN,
                                                                                                       New_Spider3,
                                                                                                       New_RaptorX,
                                                                                                       New_SSpro,
                                                                                                       New_Porter4):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])
        protein_array = np.append(protein_array, [char9])
        protein_array = np.append(protein_array, [char10])

    protein_array = protein_array[protein_array != np.array(None)]
    a = protein_array.reshape(predict_Batch_count, batch, 1, 1, active_preds[1])

    frame_row = 1
    frame_col = active_preds[1]
    channels = 1

    input_shape = (channels, frame_row, frame_col)

    model = Sequential()
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(k.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.05))

    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.05))

    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.05))

    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(10, (3, 3), activation='relu', padding="same"))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.1))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    # edit 105 to new file name
    model = load_model(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_CNN_" + active_preds[0] + ".tfl")

    final_prob = []
    Prediction_Proba = []
    for element in range(predict_Batch_count // 5):
        prob1 = model.predict(a[element])
        prediction = model.predict_classes(a[element], batch_size=batch)
        for pop in prediction:
            for box in pop:
                for count in box:
                    Prediction_Proba.append(count)
        final_prob.append(prob1)

    CNN_Consensus = ''

    for element in Prediction_Proba:
        if element == 1:
            CNN_Consensus += 'H'
        elif element == 2:
            CNN_Consensus += 'E'
        else:
            CNN_Consensus += 'C'

    CNN_Consensus = CNN_Consensus[:len(protein_layer_predictions[-1])]

    return CNN_Consensus


def Hermes_Aggregator(active_preds, protein, structure_layer_predictions):
    k.backend.tensorflow_backend.clear_session()

    np.set_printoptions(threshold=np.inf)
    protein_array = np.array([], dtype=np.int)

    Top_Protein_Layer_Predictor_1_num, New_Top_Protein_Layer_Predictor_1 = [], []
    Top_Protein_Layer_Predictor_2_num, New_Top_Protein_Layer_Predictor_2 = [], []
    Weighted_num, New_Weighted = [], []
    MLP_num, New_MLP = [], []
    RNN_num, New_RNN = [], []
    GRU_num, New_GRU = [], []
    BiLSTM_num, New_ = [], []
    CNN_num, New_CNN = [], []

    time_steps = 50
    batch = 25
    X_feature_count = 8

    for prediction, array in zip(structure_layer_predictions,
                                 [Top_Protein_Layer_Predictor_2_num, Top_Protein_Layer_Predictor_1_num,
                                  Weighted_num, MLP_num, RNN_num, GRU_num, BiLSTM_num, CNN_num]):
        for char in prediction:
            if char == 'H':
                array.append(1)
            elif char == 'E':
                array.append(2)
            elif char == 'C':
                array.append(0)

    New_Top_Protein_Layer_Predictor_2 = np.pad(Top_Protein_Layer_Predictor_2_num, (0, 10000 - len(protein)), 'constant')
    New_Top_Protein_Layer_Predictor_1 = np.pad(Top_Protein_Layer_Predictor_1_num, (0, 10000 - len(protein)), 'constant')
    New_Weighted = np.pad(Weighted_num, (0, 10000 - len(protein)), 'constant')
    New_MLP = np.pad(MLP_num, (0, 10000 - len(protein)), 'constant')
    New_RNN = np.pad(RNN_num, (0, 10000 - len(protein)), 'constant')
    New_GRU = np.pad(GRU_num, (0, 10000 - len(protein)), 'constant')
    New_BiLSTM = np.pad(BiLSTM_num, (0, 10000 - len(protein)), 'constant')
    New_CNN = np.pad(CNN_num, (0, 10000 - len(protein)), 'constant')

    predict_Batch_count = (((80000 // time_steps) // batch) // X_feature_count)

    for char1, char2, char3, char4, char5, char6, char7, char8 in itertools.zip_longest(
            New_Top_Protein_Layer_Predictor_2, New_Top_Protein_Layer_Predictor_1,
            New_Weighted, New_MLP, New_RNN, New_GRU, New_BiLSTM, New_CNN):
        protein_array = np.append(protein_array, [char1])
        protein_array = np.append(protein_array, [char2])
        protein_array = np.append(protein_array, [char3])
        protein_array = np.append(protein_array, [char4])
        protein_array = np.append(protein_array, [char5])
        protein_array = np.append(protein_array, [char6])
        protein_array = np.append(protein_array, [char7])
        protein_array = np.append(protein_array, [char8])

    protein_array = protein_array[protein_array != np.array(None)]
    a = protein_array.reshape(predict_Batch_count, batch, time_steps, X_feature_count)

    model = k.Sequential()
    model.add(Bidirectional(LSTM(X_feature_count, return_sequences=True, stateful=True),
                            input_shape=(time_steps, X_feature_count),
                            batch_input_shape=(batch, time_steps, X_feature_count)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=True)))
    model.add(Dropout(0.15))
    model.add(k.layers.TimeDistributed(Dense(X_feature_count)))
    model.add(Dense(X_feature_count))
    model.add(Dropout(0.1))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.add(k.layers.ActivityRegularization(l1=0.0, l2=0.0))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    Dir1 = str(os.getcwd()) + '/Structure Layer Neural Network Weights - Models/'

    # edit 105 to new file name
    model = load_model(Dir1 + "New_Hermes_105_" + str(active_preds[1]) + "_" + active_preds[0] + ".tfl")

    final_prob = []
    Aggregator_Consensus = ''
    for element in range(len(a)):
        prob1 = model.predict(a[element])
        Prediction_Proba = model.predict_classes(a[element])
        for array in Prediction_Proba:
            for unit in range(len(array)):
                if array[unit] == 1:
                    Aggregator_Consensus += 'H'
                elif array[unit] == 2:
                    Aggregator_Consensus += 'E'
                else:
                    Aggregator_Consensus += 'C'

        final_prob.append(prob1)

    Aggregator_Consensus = Aggregator_Consensus[:len(protein)]

    try:
        FLAG_Wholesale_Replace, OverRide_DSSP, OverRide_identity = Hermes_WholeSale_OverRide(protein)
    except:
        FLAG_Wholesale_Replace = False

    if FLAG_Wholesale_Replace == True:
        OverRide_DSSP = OverRide_DSSP[:len(protein)]
        return DSSP_8_to_3(OverRide_DSSP)
    else:
        return DSSP_8_to_3(Aggregator_Consensus)


def Hermes_WholeSale_OverRide(protein):
    URL = "https://www.rcsb.org/pdb/search/smart.do"

    PARAMS = {
        'smartSearchSubtype_0': 'SequenceQuery',
        'sequence_0': protein,
        'searchTool_0': 'blast',
        'maskLowComplexity_0': 'yes',
        'eValueCutoff_0': '0.005',
        'sequenceIdentityCutoff_0': '96',
        'smartHomologueReduction': 'true',
        'smartHomologueReductionPercent': '95',
        'target': 'Current'}

    r = requests.post(url=URL, data=PARAMS)

    FLAG_No_Homologs = False
    FLAG_Many_Homologs = False

    with open("OverRide.txt", 'w') as mot:
        for char in r.text:
            mot.write(char)

    with open("OverRide.txt", 'r') as mot:
        lines = mot.readlines()

        for line in range(len(lines)):
            if lines[line][
               :-1] == '<div class="well well-sm text-center"><h2>No results were found matching your query</h2></div>':
                FLAG_No_Homologs = True
                break
            elif lines[line][
                 :94] == '                <p class="bg-info">Your search is an <strong>ENTITY-BASED QUERY</strong> - the' and \
                            lines[line][94:96] != ' 0 ':
                FLAG_Many_Homologs = True
                New_URL = "http://www.rcsb.org/pdb/download/download.do?doQueryIds=getchecked&qrid=" + lines[line - 41][
                                                                                                       68:-3] + "&nocache=" + \
                          lines[line - 41][68:-3]
                break
            elif lines[line].startswith(
                    'console.log("Sierra release-v1.5.5-5-g50a68de east");</script><script>function gtag(){dataLayer.push(arguments)}var logger=function(){"use strict";function n(n){if(n=JSON.stringify(n),"sendBeacon"in navigator)navigator.sendBeacon(o,n);else{var e=new XMLHttpRequest;e.open("POST",o,!1),e.setRequestHeader("Content-Type","text/plain;charset=UTF-8"),e.send(n)}}function e(e,o,a){if(!RC.isProductionServer||"false"===RC.isProductionServer)switch(e){case"error":case"warn":case"info":a?console[e](o,a)'):
                pdb_codes = [lines[line][2711:2715]]
                FLAG_One_Homolog = True
                break

            elif "observeArray:function(e){for(var t=0,n=e.length;t<n;++t)r.observe(e[t])}}}();</script><title>RCSB PDB - " in \
                    lines[line]:

                pdb_codes = [lines[line][lines[line].find(
                    "observeArray:function(e){for(var t=0,n=e.length;t<n;++t)r.observe(e[t])}}}();</script><title>RCSB PDB - ") + 104:
                lines[line].find(
                    "observeArray:function(e){for(var t=0,n=e.length;t<n;++t)r.observe(e[t])}}}();</script><title>RCSB PDB - ") + 108]]
                FLAG_One_Homolog = True
                break

            else:
                pdb_codes = []
                pass

    if FLAG_Many_Homologs == True:
        r = requests.get(url=New_URL)

        with open("OverRide_Multi.txt", 'w') as mot:
            for char in r.text:
                mot.write(char)

        with open("OverRide_Multi.txt", 'r') as mot:
            lines = mot.readlines()

            for line in range(len(lines)):
                if lines[line][
                   :109] == '            <textarea class="form-control" rows="3" id="structureIdList" name="structureIdList" placeholder="':
                    prior_pdb_codes = lines[line][138:-15]

                    pre_filtered_pdb_codes = prior_pdb_codes.split()
                    pdb_codes = []
                    for element in pre_filtered_pdb_codes:
                        new_pdb_code = ''
                        for char in element:
                            if char != '>' and char != ',':
                                new_pdb_code += char
                            else:
                                pass
                        pdb_codes.append(new_pdb_code)

    if FLAG_No_Homologs != True and len(pdb_codes) != 0:
        DSSP = []
        identity = []

        for pdb_code in pdb_codes:

            PARAMS = {
                'db': 'pdbfinder',
                'q': pdb_code}

            URL_code = "https://mrs.cmbi.umcn.nl/search?db=pdbfinder&q=" + pdb_code + "&count=10"
            r = requests.post(url=URL_code, data=PARAMS)

            with open("OverRide_PDBFinder", 'w') as outfile:
                outfile.write(r.text)

            query_length = ' Amino-Acids : ' + str(len(protein))

            with open("OverRide_PDBFinder", 'r') as infile:
                lines = infile.readlines()

                for line in range(len(lines)):
                    if lines[line].startswith(" Sequence    : ") and lines[line + 1].startswith(" DSSP        : ") and \
                            (query_length == lines[line - 2][:-1] or query_length == lines[line - 1][
                                                                                     :-1] or query_length == lines[
                                                                                                                     line - 3][
                                                                                                             :-1]):
                        base_compare = 0

                        for char in range(len(lines[line]) - 15):
                            if char <= len(protein) - 1:

                                if lines[line][15 + char] == protein[char]:
                                    base_compare += 1
                        identity.append(base_compare / len(protein) * 100)
                        print(identity)
                        for tick in range(len(identity)):
                            if identity[tick] >= 95.5:
                                DSSP.append(lines[line + 1][15:])
                                break
                            else:
                                identity.pop(-1)
                                pass

    else:
        DSSP = []
        identity = 0

    for filename in ["OverRide.txt", "OverRide_Multi"]:
        if os.path.isfile(filename):
            os.remove(filename)
        else:
            pass

    if len(DSSP) != 0 and max(identity) != 0:

        maximum = identity.index(max(identity))
        FLAG_Wholesale_Replace = True
        return FLAG_Wholesale_Replace, DSSP[maximum], identity[maximum]

    else:
        FLAG_Wholesale_Replace = False

        placeholder_DSSP = 'XX'
        placeholder_identity = 0

        return FLAG_Wholesale_Replace, placeholder_DSSP, placeholder_identity


def DSSP_8_to_3(sec_struct):
    sec_struct = sec_struct.upper()
    if len(sec_struct) > 0:
        DSSP = []
        for i in range(len(sec_struct)):
            if sec_struct[i] == "H" or sec_struct[i] == "G" or sec_struct[i] == "I":
                DSSP.append("H")
            elif sec_struct[i] == "E" or sec_struct[i] == "B":
                DSSP.append("E")
            elif sec_struct[i] == "S" or sec_struct[i] == "-" or sec_struct[i] == "C" or sec_struct[i] == " " or \
                            sec_struct[i] == "T":
                DSSP.append("C")

        print("".join(list(DSSP)))

        return ("".join(list(DSSP)))


def Search_Processed_Files(protein):
    hasher = hashlib.md5(protein.encode('ASCII'))
    hasher_14_digi = hasher.hexdigest()[:14]


    filename_14_digi = 'Hermes_Result_' + str(hasher_14_digi)
    Dir1 = str(os.getcwd()) + '/'

    prot_file = None
    for file in os.listdir(Dir1 + 'Hermes Results'):
        if file.startswith('Complete ' + filename_14_digi):
            prot_file = (os.path.join(Dir1 + "Hermes Results", file))
            break

    if prot_file == None:
        for file in os.listdir(Dir1 + 'Hermes Results'):
            if file.startswith(filename_14_digi):
                prot_file = (os.path.join(Dir1 + "Hermes Results", file))
                break
            else:
                prot_file = None
                pass

    GORIV = ''
    PHD = ''
    Spider3 = ''
    RaptorX = ''
    Porter4 = ''
    pS2 = ''
    SSpro = ''
    JPRED4 = ''
    YASPIN = ''
    NetSurfP = ''
    Weighted_Vote = ''
    MLP = ''
    RNN = ''
    GRU = ''
    BiLSTM = ''
    CNN = ''
    Aggregator = ''

    print()
    print("File to be saved under: {}".format(filename_14_digi))
    print()
    print()

    if prot_file != None:
        with open(prot_file, 'r') as infile:
            for line in infile.readlines():
                if line == 0:
                    pass
                else:
                    if line[:5] == 'GORIV':
                        for char in range(len(line)):
                            if int(line[10:13]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    GORIV += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    GORIV += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    GORIV += 'C'

                    elif line[:3] == 'PHD':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    PHD += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    PHD += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    PHD += 'C'

                    elif line[:7] == 'Spider2':
                        for char in range(len(line)):
                            if int(line[12:15]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    Spider3 += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    Spider3 += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    Spider3 += 'C'

                    elif line[:7] == 'Spider3':
                        for char in range(len(line)):
                            if int(line[12:15]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    Spider3 += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    Spider3 += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    Spider3 += 'C'

                    elif line[:7] == 'RaptorX':
                        for char in range(len(line)):
                            if int(line[12:15]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    RaptorX += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    RaptorX += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    RaptorX += 'C'

                    elif line[:7] == 'Porter4':
                        for char in range(len(line)):
                            if int(line[12:15]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    Porter4 += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    Porter4 += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    Porter4 += 'C'

                    elif line[:3] == 'pS2':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    pS2 += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    pS2 += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    pS2 += 'C'

                    elif line[:5] == 'SSpro':
                        for char in range(len(line)):
                            if int(line[10:13]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    SSpro += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    SSpro += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    SSpro += 'C'

                    elif line[:6] == 'JPRED4':
                        for char in range(len(line)):
                            if int(line[11:14]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    JPRED4 += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    JPRED4 += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    JPRED4 += 'C'

                    elif line[:6] == 'YASPIN':
                        for char in range(len(line)):
                            if int(line[11:14]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    YASPIN += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    YASPIN += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    YASPIN += 'C'

                    elif line[:8] == 'NetSurfP':
                        for char in range(len(line)):
                            if int(line[13:16]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    NetSurfP += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    NetSurfP += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    NetSurfP += 'C'

                    elif line[:13] == 'Weighted Vote':
                        for char in range(len(line)):
                            if int(line[18:21]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    Weighted_Vote += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    Weighted_Vote += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    Weighted_Vote += 'C'

                    elif line[:3] == 'MLP':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    MLP += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    MLP += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    MLP += 'C'

                    elif line[:3] == 'RNN':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    RNN += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    RNN += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    RNN += 'C'

                    elif line[:3] == 'GRU':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    GRU += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    GRU += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    GRU += 'C'

                    elif line[:6] == 'BiLSTM':
                        for char in range(len(line)):
                            if int(line[11:14]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    BiLSTM += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    BiLSTM += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    BiLSTM += 'C'

                    elif line[:3] == 'CNN':
                        for char in range(len(line)):
                            if int(line[8:11]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    CNN += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    CNN += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    CNN += 'C'

                    elif line[:6] == 'Hermes':
                        for char in range(len(line)):
                            if int(line[11:14]) != len(protein):
                                if line[char] == 'a' and line[char + 1] == 'c' and line[char + 2] == 'e' and line[
                                            char + 3] == 'n':
                                    Aggregator += 'H'
                                elif line[char] == 'g' and line[char + 1] == 'r' and line[char + 2] == 'e' and line[
                                            char + 3] == 'e' and line[char + 4] == 'n':
                                    Aggregator += 'E'
                                elif line[char] == 's' and line[char + 1] == 't' and line[char + 2] == 'a' and line[
                                            char + 3] == 'l' and line[char + 4] == 'k':
                                    Aggregator += 'C'

    protein_layer_saved = 0
    joined_names = ''
    for prediction, predictor in zip([GORIV, PHD, NetSurfP, pS2, JPRED4, YASPIN, Spider3, RaptorX, SSpro, Porter4],
                                     ['GORIV', 'PHD', 'NetSurfP', 'pS2', 'JPRED4', 'YASPIN', 'Spider3', 'RaptorX',
                                      'SSpro', 'Porter4']):
        if prediction != '':
            joined_names += predictor + '_'
            protein_layer_saved += 1

    saved_structure_list_flag = False
    saved_aggregator_flag = False
    structure_layer_saved = 0
    for element in [Weighted_Vote, MLP, RNN, GRU, BiLSTM, CNN]:
        if element != '' and len(element) == len(protein):
            structure_layer_saved += 1

    if structure_layer_saved == 6:
        saved_structure_list_flag = True
    if Aggregator != '':
        saved_aggregator_flag = True

    active_preds = [joined_names[:-1], protein_layer_saved, protein]

    return hasher_14_digi, prot_file, [GORIV, PHD, NetSurfP, pS2, JPRED4, YASPIN, Spider3, RaptorX, SSpro, Porter4], [
        Weighted_Vote, MLP, RNN, GRU, BiLSTM,
        CNN], Aggregator, saved_structure_list_flag, saved_aggregator_flag, active_preds


def Save_Prot_Predict(protein, prot_name):
    hasher = hashlib.md5(protein.encode('ASCII'))
    hasher = hasher.hexdigest()[:14]

    filename = 'Complete Hermes_Result_' + hasher

    prot_name += '.txt'

    Dir1 = str(os.getcwd()) + '/Hermes Results/'

    with open(prot_name, 'r') as infile:
        with open(Dir1 + filename, 'w') as outfile:
            outfile.write(protein + '\n')
            for line in infile.readlines():
                outfile.write(str(line))

    if os.path.isfile("temp.txt"):
        os.remove("temp.txt")
    else:
        pass

    return filename
