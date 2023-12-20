## Run script to train the model

```bash
docker build -t homevision .
docker run -ti --rm -v $(pwd)/app/model:/app/model  homevision
```

## Run script to predict

```bash
docker build -t homevision .
docker run -it --rm -v $(pwd)/app/model:/app/model homevision python evaluate_model.py
```




