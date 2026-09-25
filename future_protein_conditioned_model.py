# Research extension template
# A true protein-conditioned model should be trained on multiple protein-ligand targets:
#
# protein sequence/embedding + ligand representation -> measured activity
#
# Do NOT train a protein-conditioned model using only NDM-1 and then call it general SBDD.
# Suitable future sources include ChEMBL target/activity records and pretrained protein
# representations. Validate with target/scaffold-aware splits.
