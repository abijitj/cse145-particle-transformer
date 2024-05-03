import uproot
import os

example_file = "./retrain_test_1/JetClass/Pythia/test_20M/HToBB_100.root"

# Load the content from the file
tree = uproot.open(example_file)['tree']

# Display the content of the "tree"
tree.show()