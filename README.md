# Sentiment Analysis using LSTM
Implementing the towards data science article of the same name (https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948) in TensorFlow 2.0 using the IMDB review dataset from http://ai.stanford.edu/~amaas/data/sentiment.

# How to run
- Download the GitHub repository.
- Run `conda sentimentlstm create -f environment.yml` to create the conda environment.
- Activate the environment with `conda activate sentimentlstm`.
- Start the jupyter notebook with `jupyter notebook sentimentLSTM.ipynb`.
- Execute the notebook.
- Play around and experiment with it. 

# Main Differences
The first and major difference is the data loading. I integrated the process of downloading and formatting the data into the notebook while in the article it was a bit omitted (probably to keep it more concise). Also, I am loading the data into dataframes because I think it makes the code easier to read. 

The other difference is the train/val split. I did not use a final test set to evaluate performance because the objective of this project was to sucessfully translate the code into the other framework and also no hyperparameter tuning is done so the validation score can be used as a final test score. 

The last change is the again the for sake of readability omitted training loop that I added. It is as mentioned in the article very standard. 

# Conclusion 
Tensorflow 2.0 looks very promising. It takes all the things I and most other people like about PyTorch and integrates it into Tensorflow while keeping what made it stand out so far like Tensorboard. 
