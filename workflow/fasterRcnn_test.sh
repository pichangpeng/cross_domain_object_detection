for name in {day,day_fakeNight,day_night}
do
    for i in {1..5}
    do
           python3 main/fasterRcnn_test.py --batchSize 3  --imagesRoot data/images/test/day data/images/test/night --labelsRoot data/labels/test/day.json data/labels/test/night.json --fasterRcnn_weight output/weight/fasterRcnn_${name}_${i}.pth --train_name ${name} --n_experiment ${i} --test_name day_night
    done
done


for name in {day,fakeNight,day_fakeNight,night}
do
    for i in {1..5}
    do
           python3 main/fasterRcnn_test.py --batchSize 3  --imagesRoot data/images/test/night --labelsRoot data/labels/test/night.json --fasterRcnn_weight output/weight/fasterRcnn_${name}_${i}.pth --train_name ${name} --n_experiment ${i} --test_name night
    done
done
