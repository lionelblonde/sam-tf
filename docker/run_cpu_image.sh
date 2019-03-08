cd ..

docker run -i -t --rm -v $(pwd)/MJKEY:/MJKEY \
                      -v $(pwd)/DEMOS:/code/DEMOS \
                      -v $(pwd)/data:/code/data \
                      -v $(pwd)/imitation:/code/imitation \
                      -v $(pwd)/launchers:/code/launchers docker-sam-tf-cpu:latest bash
