#! /bin/bash


for seed in 166; do
  echo $seed
  mkdir data_${seed}

  th=`ls -1 ../seeds/${seed}/test* | wc | awk '{print($1-1)}'`

  for i in `seq ${th}`; do
    echo $i
    java -jar qt.jar -exec "python ../src/main2.py" -seed ${seed} -folder ../seeds/ -targetHour ${i} -train
    mv bar_${i}.csv data_${seed}/bbar_${seed}_bar_${i}.csv
  done

done
