# Machine Learning Fundamentals - Project Checkpoint

## Experiments and Results

### Dylan's Experiments

## Results Visualization

## Difficulties Encountered

I encountered issues with how slow my kNN model would take to compute for rapid debugging and testing, but I solved that by just filtering the dataset and only using the whole dataset for when the python script was finished. Another difficulty would be knowing what X or Y axis to use for kNN due to multiple features being tested. I am also not sure if I need to provide more valuable data other than accuracy or balanced accuracy. While experimenting, I encountered an oversight which would be that I did not normalize my feature values, so values such as NA sales dominated the model. I changed it so that all of the features are normalized.

## Remaining Work

1. Create a visualization of kNN boundaries. This will help me understand how the model separates genres in feature space, and reveal whether genre clusters seperable or entangled.
2. Develop another experiment with a Decision Tree Model. 
3. I plan to compare the performance of my kNN classifier to a Decision Tree model using the same features, to test whether a model with built-in feature weighting can outperform kNN.
4. I plan to run models excluding individual features, like removing year date as a feature, to evaluate which features contribute most to predictive power. This helps identify noise vs signal in my dataset.