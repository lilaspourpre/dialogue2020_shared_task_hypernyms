all: codalab.zip

codalab/scoring_program.zip: scoring_program/*
	cd scoring_program && zip -r ../codalab/scoring_program.zip * && cd ..

codalab/reference_data.zip:
	zip -j codalab/reference_data.zip dataset/dev_synsets.tsv dataset/ruwordnet.db

codalab.zip: codalab/* codalab/scoring_program.zip codalab/reference_data.zip
	zip -j codalab.zip codalab/* && rm codalab/scoring_program.zip codalab/reference_data.zip
