# [CS188: Project 3 - RL](https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html)
> Pacman seeks reward. 
> 
> Should he eat or should he run? 
> 
> When in doubt, Q-learn.

```python
class MarkovDecisionProcess
```

## Crawler
`python crawler.py`

## Pacman(Exact Q-Learning)
`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid`

## Pacman(Approximate Q-Learning)
- `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l mediumGrid`
- `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 53 -l mediumClassic` (**NOTE**: may take a few seconds to train)
- `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 100 -n 103 -l smallClassic`

