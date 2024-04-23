for radius in .0 .0003 .0001 .001 
do
    for tradeoff in 0 .3 
    do 
        for align in .3 1 3
        do
            sbatch run.sh $radius $tradeoff $align
        done
    done
done