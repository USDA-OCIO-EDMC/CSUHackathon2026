**How To Use Our Project:**

1: Run goobie6000.ipnyb first. This will load geospatial data and classify land plots based on the likelihood that is is a cornfield. Results on accuracy will be output in the command line.
2: Run yield_predictions.ipnyb. This will generate regression data using the model created in goobie6000.ipnyb.

*Note* 

Do not run train.py. It is only used for concurrency inside SagemakerAI to speed up goobie, and doesn't provide any results on its own.

##Additional Info##
We used Prithvi-EO for our model. Since it is too large to upload to GitHub, you may access a zip file of the model here: https://drive.google.com/file/d/1iS90r5DQrehxiYXljxRHTpyudAYEMgyg/view?usp=sharing
