# Hermes Preprint: https://www.biorxiv.org/content/10.1101/640656v1

~ Due to GitHub file size limits, this repository contains a version of Hermes without the mechanism to deal with individual Protein layer server crashes. The full version is avalible as a zipped folder from Google Drive: https://drive.google.com/open?id=1sy_DXgULBPtS4TAYqY2fyF9TnbqkOpDj

1. Download the master branch
2. Run the file "Houston.py"
3. Results are given as output to the console

# Dependencies

Python Version: 3.6.2

Required Python Packages

Name: Tensorflow
Version: 1.5.0
URL Access: https://pypi.org/project/tensorflow/1.5.0/

Name: beautifulsoup4
Version: 4.6.0
URL Access: https://pypi.org/project/beautifulsoup4/4.6.0/

Name: requests
Version: 2.17.3
URL Access: https://pypi.org/project/requests/2.17.3/

Name: imaplib2
Version: 2.45.0
URL Access: https://pypi.org/project/imaplib2/2.45.0/

Name: numpy
Version: 1.13.1
URL Access: https://pypi.org/project/numpy/1.13.1/

Name: Keras
Version: 2.1.3
URL Access: https://pypi.org/project/Keras/2.1.3/

# Files Provided
1. Houston.py
• The laucher file

2. Hermes_Main.py
• The core pipeline, intitated by the launcher

3. Sample_DNA_Seqs_For_Hermes.Fa
• Multiple DNA sequences, given in FASTA format, to provide an initial run

4. triplet.txt
• The codon table as a text file, which is used internally

5. 1Letter.txt
• The one-letter amino acid codes as a text file, again used internally

6. /Structure Layer Neural Network Weights - Models/Structure_Layer_BiLSTM.tfl
• The Bidirectional LSTM (BiLSTM) classifier weights & biases of the Structure layer

7. /Structure Layer Neural Network Weights - Models/Structure_Layer_CNN.tfl
• The Convolutional Neural Network (CNN) classifier weights & biases of the Structure layer

8. /Structure Layer Neural Network Weights - Models/Structure_Layer_GRU.tfl
• The Gated Recurrent Unit (GRU) classifier weights & biases of the Structure layer

9. /Structure Layer Neural Network Weights - Models/Structure_Layer_MLP.pickle
• The Multilayer Perceptron (MLP) classifier weights & biases of the Structure layer

10. /Structure Layer Neural Network Weights - Models/Structure_Layer_RNN.tfl
• The Recurrent Neural Network (RNN) classifier weights & biases of the Structure layer

11. /Structure Layer Neural Network Weights - Models/Hermes_Aggregator_Model.tfl
• The final aggregator network weights & biases, a BiLSTM classifier

12. /Hermes Results/Hermes_Result_d23481bb7250fb.txt
• A text file, serving as a single example within the cashe system, that is used to save protein layer predictions

13. /Hermes Results/Hermes_Result_facff3fc114cb2.txt
• Another text file, serving as a second example within the cashe system, that is used to save protein layer predictions

14. /Evaluation Files/JPRED Blind Test/All JPRED Blind Test Predictions w SOV Scores Redundancy Reduced.txt
• A text file of all JPRED Blind test proteins, with all predictions, respective Q3's, and the experimentally determined DSSP-classified structure

15. /Evaluation Files/JPRED Blind Test/All JPRED Blind Test NEFF & Identity Values
• A text file of all NEFF & identity values for all the JPRED Blind test proteins

16. /Evaluation Files/CASP11/All CASP11 Predictions w SOV Scores Redundancy Reduced.txt
• A text file of all CASP11 proteins, with all predictions, respective Q3's, and the experimentally determined DSSP-classified structure

17. /Evaluation Files/CASP11/All CASP11 NEFF & Identity Values
• A text file of all NEFF & identity values for all the CASP11 proteins

18. /Evaluation Files/CASP12/All CASP12 Predictions w SOV Scores Redundancy Reduced.txt
• A text file of all CASP12 proteins, with all predictions, respective Q3's, and the experimentally determined DSSP-classified structure

19. /Evaluation Files/CASP12/All CASP12 NEFF & Identity Values
• A text file of all NEFF & identity values for all the CASP12 proteins

20. /Evaluation Files/CASP13/All CASP13 Predictions w SOV Scores Redundancy Reduced.txt
• A text file of all CASP13 proteins, with all predictions, respective Q3's, and the experimentally determined DSSP-classified structure

21. /Evaluation Files/CASP13/All CASP13 NEFF & Identity Values
• A text file of all NEFF & identity values for all the CASP13 proteins

22. /Evaluation Files/7-Fold Cross Validation/7-Fold CV Model 1.tfl-7-Fold CV Model 7.tfl
• Seven files, each a set of weights & biases for the respective cross-validation model

23. /Evaluation Files/Training/All Training Predictions w SOV Scores Redundancy Reduced.txt
• A text file of all Training proteins, with all predictions, respective Q3's, and the experimentally determined DSSP-classified structure
