# Homemade KMeans Clustering
This repo is my own attempt at re-creating the KMeans Clustering algorithm from scratch, in a simple way, using python solely. 

By no means does this module attempt to improve or implement a better or more efficient version of the algorithm than the ones already out there. This is my rather *lame* attempt at re-creating it with as little help as possible from external libraries (e.g.., we create our own squared euclidean distance function, rather than using `numpy` - definitely a faster way).

## So you may be asking yourself... why?

There is no why. This is a product of my boredoom.

# Example Run

After spinning up the container, or your preferred environment, go to `/notebooks` and you can run examples there.

# The Algorithm

Following the algorithm defined in [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), we define the following algorithm:

1. For a given cluster assignment $C$, the total cluster variance is minimized with respect to $m_{1}, ..., m_{K}$ yielding the means of the currently assigned clusters.

2. Given a current seat of means $m_{1}, ..., m_{K}$, the total cluster variance is minimized by assigning each observation to the closest (current) cluster mean. That is,

$$ C(i) = argmin_{1 \leq k \leq K} ||x_{i} - m_{k}||^2$$

3. Repeat steps 1 and 2 until the assignments do not change

Where the Total Cluster Variance is defined as follows:

$$ \min_{C, {m_k} ^K_1 } \sum_{k=1}^{K} N_{k} ~ \sum_{C(i) = k} ||x_{i} - m_{k}||^2 $$

where

$$ N_k = \sum_{i = 1}^N I(C(i) = k) $$

# Provided Docker Environment

You can try this using the Jupyter Notebook official Docker image.

```bash
docker-compose down && docker-compose up --build -d
```

Go to localhost:8000 or 127.0.0.1:8000 and start the notebook. The token (playground) is defined in the `Dockerfile`.
