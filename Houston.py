import time
import tensorflow as tf
import itertools
from Hermes_Main import Sequence


''' Below is the fasta file containing sequences to predict '''
sequence = 'Sample_DNA_Seqs_For_Hermes.fa'

# Below are email addresses for protein layer predictions to be received
email_address_list = ['audibleabc1@gmail.com', 'audibleabc2@gmail.com', 'audibleabc3@gmail.com',
    'protprotprot000@gmail.com', 'protprotprot222@gmail.com', 'protprotprot444@gmail.com', 'protprotprot666@gmail.com',
    'protprotprot777@gmail.com', 'protprotprot888@gmail.com', 'protprotprot999@gmail.com',
    'protprotprotprot2222@gmail.com', 'protprotprotprot3333@gmail.com', 'requirementsrequired@gmail.com']

count = 1

with open(sequence, 'r') as seq_holder:
    for seq, email_address in zip(seq_holder.read().split('>'), itertools.cycle(email_address_list)):
        individual_seq_file = 'test1' + seq[:8] + '.txt'
        with open(individual_seq_file, 'w') as infile:
            infile.write(seq)

        try:
            start = time.time()



            query = Sequence(email_address, individual_seq_file)

            query.Transcribe()

            query.Translate()

            query.Protein_Layer()

            query.Structure_Layer()

            query.Final_Hermes()



            print()
            print()
            print()
            print()
            print("Total Number of Proteins Analysed on This Run: {}".format(count))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print()
            print()
            print()
            print()
            tf.reset_default_graph()
            tf.keras.backend.clear_session()

        except:

            print()
            print()
            print()
            print()
            print(
                "||||||||||||||||||| AN EXCEPTION HAS OCCURED, SKIPPING TO THE NEXT SEQ|||||||||||||||||||")
            print()
            print()
            print()
            print()
            pass

        count += 1
#