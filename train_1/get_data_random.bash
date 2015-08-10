#! /bin/bash


#for seed in `ls -1 ../seeds`; do
for seed in 105 115 129 138 143 148 15  152 156 166 17 113 116 13  142 147 151 155 163 169 171; do
  echo $seed
  mkdir data_${seed}

  eh=`ls -1 ../seeds/${seed}/test* | wc | awk '{print($1-1)}'`

  # 20 randoms from each
  for i in `seq 50`; do
    th=`echo $RANDOM % $eh | bc`
    echo $i $th
    java -jar qt.jar -exec "python ../src/main2.py" -seed ${seed} -folder ../seeds/ -targetHour ${th} -train
    mv bar_${th}.csv data_${seed}/bbar_${seed}_bar_${th}.csv
  done

done
