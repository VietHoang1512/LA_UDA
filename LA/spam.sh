for radius in .0001 .0003 .00003
do
    for tradeoff in  .1 .5 1
    do 
        for align in 0. .3 1 3
        do
            sbatch run.sh $radius $tradeoff $align
        done
    done
done