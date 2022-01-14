NUM_MODELS := 10


#Generate inference results 
model_list.log labelled_img_list.log: raw_data.json
	mkdir -p logs; \
	mkdir -p data; \
	mkdir -p models; \ 
	touch logs/model_list.log; \
	touch logs/labelled_img_list.log; \
	for i in {1..$(NUM_MODELS)}
	do
		echo "Generating result for model $i / $(NUM_MODELS)" >> logs/run.log; \
		python3 model.py raw_data.json "model_$i"; \
	done

#Assimilate results
assimilated_results.log: labelled_img_list.log
	touch $@; \
	python3 discuss.py $^ $@ >> logs/run.log

#Use ground truth to teach models
loss.log: model_list.log assimilated_results.log
	touch loss.log; \ 
	python3 teach.py model_list.log assimilated_results.log; \
	echo $^ >> debug.log

clean: 
	rm -rf logs
	rm -rf data