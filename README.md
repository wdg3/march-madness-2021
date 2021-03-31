# March Madness 2021
This project trains an AdaBoost model to output March Madness bracket predictions using data provided by Kaggle.

## Dependencies
- Python3
- pandas
- sklearn

## Directions
First, go to Kaggle to download the [dataset](https://www.kaggle.com/c/ncaam-march-mania-2021/data). Either save this in a directory named /data/march-madness or edit <code>data-processing.py</code> to reflect the data directory.

Then run <code>python main.py</code>. Done! Predictions for each game will be saved to <code>[data directory]/MGamePreds.csv</code>, where for each possible game, there is a line with the two teams playing and the probability that the first team wins.
