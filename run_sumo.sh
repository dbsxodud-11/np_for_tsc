###### 0. Experiment setup
exp_id="debug"


###### 1. collect_data
for task in sumo_3by3 sumo_4by4 sumo_5by5 sumo_hangzhou sumo_manhattan
do
 python collect_data.py --task ${task}
done


####### 2. train 
for task in sumo_3by3 sumo_4by4 sumo_5by5 sumo_hangzhou sumo_manhattan
do
 python train.py --task ${task} --model anp --exp_id $exp_id
done

####### 3. test
for task in sumo_3by3 sumo_4by4 sumo_5by5 sumo_hangzhou sumo_manhattan
do
  for i in {0..9}
  do
    for model in anp
    do
      python test.py --task ${task} --model ${model} --exp_id $exp_id --scenario_id $i
    done
  done
done
