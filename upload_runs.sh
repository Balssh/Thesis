#!/bin/bash

BASE_PATH="runs/ClassicControlRuns/SeparateNNs"
for path in $(ls $BASE_PATH); do
	echo "Uploading directory $path"
	tensorboard dev upload --logdir "$BASE_PATH/$path" --name "SeparateNN/$path"
	echo "---------------------------------"
done
