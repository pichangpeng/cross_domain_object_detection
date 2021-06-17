# for a in {4..5}
# do
#        python3 main/fasterRcnn_train.py --n_epochs 50 --batchSize 6  --experiment_name day --imagesRoot data/images/train/day --labelsRoot data/labels/train/day.json --lr 0.000002 --n_experiment $a
# done


# for a in {3..5}
# do
#        python3 main/fasterRcnn_train.py --n_epochs 50 --batchSize 6  --experiment_name day_fakeNight --imagesRoot data/images/train/day output/images/cycleGAN/1_720_1280_1/fake_night --labelsRoot data/labels/train/day.json --lr 0.000002 --n_experiment $a
# done

# for a in {2..3}
# do
#        python3 main/fasterRcnn_train.py --n_epochs 50 --batchSize 6  --experiment_name day_night --imagesRoot data/images/train/day data/images/train/night --labelsRoot data/labels/train/day.json data/labels/train/night.json --lr 0.000002 --n_experiment $a
# done


# for a in {5..6}
# do
#        python3 main/fasterRcnn_train.py --n_epochs 50 --batchSize 6  --experiment_name fakeNight --imagesRoot output/images/cycleGAN/1_720_1280_1/fake_night --labelsRoot data/labels/train/day.json --lr 0.000002 --n_experiment $a
# done


for a in {4..5}
do
       python3 main/fasterRcnn_train.py --n_epochs 50 --batchSize 6  --experiment_name night --imagesRoot data/images/train/night --labelsRoot data/labels/train/night.json --lr 0.000002 --n_experiment $a
done