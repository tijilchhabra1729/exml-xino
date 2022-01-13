# exml-xino




We have used CNN model for image classification. Our architecture is one convulational layer architecture with 32 3*3 filters and max pooling with 2*2, stride is default as 1, padding is kept same to ensure that output has same size as input. 
Final output layer has 8 outputs as we have 8 distinct classes of logos. In the final layer, activation used is Softmax. This is used as predicted probabilities of corresponding logo classes. 
Our one layer network uses the efficient “Adam” gradient descent optimization algorithm (this alogorithm in keras uses default learning rate of 0.001) and “categorical_crossentropy” of keras is used as  logarithmic loss function.
