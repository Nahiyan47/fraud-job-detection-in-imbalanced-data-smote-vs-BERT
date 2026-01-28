—Online job portals have become a popular target
for scammers who post fake job listings to exploit job seekers.
Identifying these fraudulent postings is tricky, especially when
genuine listings vastly outnumber the fake ones. In this study,
we worked with a dataset of over 18,000 job postings where
only about 800 were fraudulent, making it a highly imbalanced
classification problem. We tried three different approaches to
tackle this. First, we trained a logistic regression model using TF
IDF features and class weights without any oversampling. Second,
we applied SMOTE to balance the training data before fitting
the same logistic regression model. Third, we fine-tuned a BERT
transformer model on the original imbalanced data without
any resampling technique. Our results were quite interesting.
Both logistic regression models, with and without SMOTE,
showed similar performance with recall around 86-87% but poor
precision of only 22-25% for detecting fraudulent posts. SMOTE
did not provide any meaningful improvement over simple class
weighting. However, BERT told a different story. It achieved
86% recall with 96% precision, resulting in an F1-score of 0.91
compared to just 0.35-0.39 for the logistic models. What we found
is that for text classification tasks involving imbalanced datasets,
BERT handles the minority class much better without needing
synthetic oversampling. Its pretrained language understanding
allows it to pick up subtle patterns from limited examples. This
makes transformer-based models a strong choice when dealing
with imbalanced textual data where traditional techniques like
SMOTE offer little help
