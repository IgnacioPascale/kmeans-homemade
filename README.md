# Homemade KMeans Clustering
This repo is my own attempt at re-creating the KMeans Clustering algorithm from scratch, in a simple way, using python solely. 

By no means does this module attempt to improve or implement a better or more efficient version of the algorithm than the ones already out there. This is my rather *lame* attempt at re-creating it with as little help as possible from external libraries (e.g.., we create our own squared euclidean distance function, rather than using `numpy` - definitely a faster way).

**So you may be asking yourself... why?**

There is no why. This is a product of my boredoom.

## Example Run

After spinning up the container, or your preferred environment, go to `/notebooks` and you can run examples there.

## Provided Docker Environment

You can try this using the Jupyter Notebook official Docker image.

```bash
docker-compose down && docker-compose up --build -d
```

Go to localhost:8000 or 127.0.0.1:8000 and start the notebook.
