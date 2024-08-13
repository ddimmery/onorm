# Online Normalization (onorm)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code-of-conduct.md)
[![ci](https://github.com/ddimmery/onorm/actions/workflows/ci.yml/badge.svg)](https://github.com/ddimmery/onorm/actions/workflows/ci.yml)
![PyPI](https://img.shields.io/pypi/v/onorm)


```python
import numpy as np
import pandas as pd
from plotnine import aes, geom_point, ggplot, theme_minimal
```


```python
n = 100
d = 2

X = np.random.normal(size=(n, d))
df = pd.DataFrame(X, columns=["X1", "X2"])

ggplot(df, aes("X1", "X2")) + geom_point() + theme_minimal()
```


    
![png](README_files/README_2_0.png)
    

