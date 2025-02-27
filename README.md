### Our Data is pretty Big. Oh and we are Group 10

We chose the following binary classification task:

* [IMDB Project](imdb/) - learn to identify highly rated movies

Consult the [project page on Canvas](https://canvas.uva.nl/courses/49690/pages/projects) for detailed instructions on the scope and grading of the projects.

### Submitting predictions

#### How to generate predictions

Each project contains two files `validation_hidden.csv` and `test_hidden.csv`, with the data for which your ML pipeline has to create predictions.

In order to submit your predictions, you need to create two text files (one for the validation set and one for the test set). Each line in these files
must consist of either the string `True` or the string `False`, which denote the predicted class for the corresponding data item in the validation or test files.

#### The submission server

In order to submit predictions for your team, you have to use our [submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/). The access credentials for the submission server will be given out by the TAs in next week's lab.

For each submission, the submission server will compute the accuracy on the validation set and the test set. However, only the accuracy on the validation set will be shown (and used to generate the leaderboard).

For each project, there is a _random-baseline_ submission, which shows the accuracy achieved by random guessing, and a _ta-baseline_ submission, which shows the accuracy of a minimal submission created by one of the TAs.

Each team can submit only five times per day.

 * [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)
