all: codalab.zip

codalab/scoring_program.zip: scoring_program/*
	cd scoring_program && zip -r ../codalab/scoring_program.zip * && cd ..

codalab/public_nouns.zip:
	zip -j codalab/public_nouns.zip dataset/public/nouns_public_synsets.tsv

codalab/public_verbs.zip:
	zip -j codalab/public_verbs.zip dataset/public/verbs_public_synsets.tsv

codalab/private_nouns.zip:
	zip -j codalab/private_nouns.zip dataset/private/nouns_private_synsets.tsv

codalab/private_verbs.zip:
	zip -j codalab/private_verbs.zip dataset/private/verbs_private_synsets.tsv 

codalab.zip: codalab/* codalab/scoring_program.zip codalab/private_nouns.zip codalab/private_verbs.zip codalab/public_nouns.zip codalab/public_verbs.zip
	zip -j codalab.zip codalab/* && rm codalab/scoring_program.zip codalab/private_nouns.zip codalab/private_verbs.zip codalab/public_nouns.zip codalab/public_verbs.zip
