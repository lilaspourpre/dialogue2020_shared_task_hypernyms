all: competition.zip

competition/scoring_program.zip: scoring_program/*
	cd scoring_program && zip ../competition/scoring_program.zip * && cd ..

competition.zip: competition/* competition/scoring_program.zip #competition/dev_data.zip competition/test_data.zip
	cd competition && zip ../competition.zip * && cd ..

.PHONY: solutions

solutions: active-dict-solution.zip bts-rnc-solution.zip

active-dict-solution.zip: ../data/active-dict/*
	cp ../data/active-dict/test-solution.csv . && \
		cp ../data/active-dict/public.txt . && \
		echo "This is the authoritative result." > metadata && \
		zip active-dict-solution.zip metadata test-solution.csv public.txt && \
		rm metadata test-solution.csv public.txt

bts-rnc-solution.zip: ../data/bts-rnc/*
	cp ../data/bts-rnc/test-solution.csv . && \
		cp ../data/bts-rnc/public.txt . && \
		echo "This is the authoritative result." > metadata && \
		zip bts-rnc-solution.zip metadata test-solution.csv public.txt && \
		rm metadata test-solution.csv public.txt

wiki-wiki-solution.zip: ../data/wiki-wiki/*
	cp ../data/wiki-wiki/test-solution.csv . && \
		cp ../data/wiki-wiki/public.txt . && \
		echo "This is the authoritative result." > metadata && \
		zip wiki-wiki-solution.zip metadata test-solution.csv public.txt && \
		rm metadata test-solution.csv public.txt
