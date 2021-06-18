for name in {day,day_fakeNight,day_night}
do
    for i in {1..5}
    do
        python3 model/metric.py --model_name fasterRcnn/${name}_${i}_to_day_night
    done
done



for name in {day,fakeNight,day_fakeNight,night}
do
    for i in {1..5}
    do
        python3 model/metric.py --model_name fasterRcnn/${name}_${i}_to_night
    done
done