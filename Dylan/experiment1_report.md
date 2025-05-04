# Machine Learning Fundamentals - Project Checkpoint

## Experiments and Results

### Dylan's Experiments

I hypothesized that genre classification could be reliably predicted using sales, release year, and publishing company, due to inherent patterns in how different genres perform over time and across markets. I will be using a kNN model and a Decision Tree model to perform this.

In my first experiment, I used a kNN model to attempt to predict game genres based on their sales, publisher, and release year. I used an 80/20 split, with 13062 training samples and 3265 predictions. After testing several variations of k, I found that k=10 gave the highest balanced accuracy out of 1-100 k values. I found that my Accuracy was 35.8% and Balanced Accuracy was 30.62%. There are 12 genres, so randomly guessing a genre would be an 8.33% chance, which would be compared to the Balance Accuracy calculation, since it accounts for the variance in genre sample sizes as there are an overwhelming amount of games that are Action compared to Puzzle for example. This implies that the kNN model is able to predict the genre by 3.6x more accurately than a random guess. A key takeaway from these findings is that despite a substantial increase in accuracy for predicting game genre based on several features, it is still very unreliable to predict a game genre purely by its release time, publishing company, and sales success. This could be due to limited predictive power of the chosen features. I also found that lower k values overfit to noisy local neighbors, while higher k values oversmoothed class distinctions. k=10 balanced these effects. The substantial increase in accuracy is most likely due to companies, like Nintendo, finding a successful niche such as platforming. An example would be that a high sales, Nintendo game in any year would most likely be a platformer or another genre that they specialize in.

## Results Visualization

| k Value | Accuracy | Balanced Accuracy |
|--------:|---------:|------------------:|
|       1 |   0.3207 |           0.2914  |
|       3 |   0.3314 |           0.2961  |
|       5 |   0.3415 |           0.2959  |
|     *10 |  *0.3580 |          *0.3062  |
|      25 |   0.3424 |           0.2798  |
|      50 |   0.3332 |           0.2564  |
|     100 |   0.3139 |           0.2259  |

| Genre         | Count |
|---------------|------:|
| Action        |  3253 |
| Sports        |  2304 |
| Misc          |  1710 |
| Role-Playing  |  1471 |
| Shooter       |  1282 |
| Adventure     |  1276 |
| Racing        |  1226 |
| Platform      |   876 |
| Simulation    |   851 |
| Fighting      |   836 |
| Strategy      |   671 |
| Puzzle        |   571 |

## Difficulties Encountered

I encountered issues with how slow my kNN model would take to compute for rapid debugging and testing, but I solved that by just filtering the dataset and only using the whole dataset for when the python script was finished. Another difficulty would be knowing what X or Y axis to use for kNN due to multiple features being tested. I am also not sure if I need to provide more valuable data other than accuracy or balanced accuracy. While experimenting, I encountered an oversight which would be that I did not normalize my feature values, so values such as NA sales dominated the model. I changed it so that all of the features are normalized.

## Remaining Work

1. Create a visualization of kNN boundaries. This will help me understand how the model separates genres in feature space, and reveal whether genre clusters seperable or entangled.
2. Develop another experiment with a Decision Tree Model. 
3. I plan to compare the performance of my kNN classifier to a Decision Tree model using the same features, to test whether a model with built-in feature weighting can outperform kNN.
4. I plan to run models excluding individual features, like removing year date as a feature, to evaluate which features contribute most to predictive power. This helps identify noise vs signal in my dataset.