all: codalab.zip

codalab/scoring_program.zip: scoring_program/*
	cd scoring_program && zip -r ../codalab/scoring_program.zip * && cd ..

codalab/reference_data_public_nouns.zip:
	zip -j codalab/reference_data_public_nouns.zip dataset/public/nouns_public_synsets.tsv dataset/ruwordnet.db

codalab/reference_data_public_verbs.zip:
	zip -j codalab/reference_data_public_verbs.zip dataset/public/verbs_public_synsets.tsv dataset/ruwordnet.db

codalab/reference_data_nouns.zip:
	zip -j codalab/reference_data_nouns.zip dataset/public/nouns_public_synsets.tsv dataset/private/nouns_private_synsets.tsv dataset/ruwordnet.db

codalab/reference_data_verbs.zip:
	zip -j codalab/reference_data_verbs.zip dataset/public/verbs_public_synsets.tsv dataset/private/verbs_private_synsets.tsv dataset/ruwordnet.db

codalab.zip: codalab/* codalab/scoring_program.zip codalab/reference_data_nouns.zip codalab/reference_data_verbs.zip codalab/reference_data_public_nouns.zip codalab/reference_data_public_verbs.zip
	zip -j codalab.zip codalab/* && rm codalab/scoring_program.zip codalab/reference_data_nouns.zip codalab/reference_data_verbs.zip codalab/reference_data_public_nouns.zip codalab/reference_data_public_verbs.zip
