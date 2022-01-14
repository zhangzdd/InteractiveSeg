NUM_MODELS:=3
SHELL:=/bin/bash
ROUND:=1
all: loss.log

#Generate inference results ßs
labelled_img_list.log: raw_data.json
	mkdir -p logs; \
	mkdir -p data; \
	mkdir -p data/initial_label; \
	mkdir -p models; \
	mkdir -p models/round_$(ROUND)
	touch logs/labelled_img_list.log; \
	for i in {1..$(NUM_MODELS)}; \
	do \
		echo $$i >> logs/debug.log; \
		echo "Generating result for model $$i / $(NUM_MODELS)" >> logs/run.log; \
		python3 forward_pass.py raw_data.json "model_$$i" $(ROUND); \
	done 
	ls data/initial_label >> logs/labelled_img_list.log

#Make a list of models
model_list.log: labelled_img_list.log
	touch logs/model_list.log; \
	ls models/round_$(ROUND) >> logs/model_list.log

#Assimilate results
assimilated_results.log: labelled_img_list.log
	mkdir -p data/variance; \
	touch logs/$@; \
	python3 discuss.py $^ $@ >> logs/run.log

#Use ground truth to teach models
loss.log: model_list.log assimilated_results.log labelled_img_list.log
	touch logs/loss.log; \
	python3 teach.py $^; \
	echo "Round_$(ROUND) Instruction Ended" >> logs/debug.log

clean: 
	rm -rf logs
	rm -rf data
	rm -rf models