Team 15 prompt 2
We made a cloud-native satellite image pipeline. Which finds HLS satellite imagery, pulls only usable images from the database, from those images we get vegetation health numbers, creates preview images, and try to run Prithvi feature extraction. It has true yield predictions which we added last, before that we had no trained yield model attached. We have what we plug into a tried model’s neural network but we don’t actually make it guess, but we worked on that until our time was up.

Our imports let Python interact with the operating system, like setting cloud-access environment variables,  clean up memory after large satellite images are processed, and makes our file and folder paths cleaner and safer than just using strings. GeoPandas is used to read map boundary files like GeoJSON and pandas make our csv’s and tables. Torch loads PyTorch, used to run the Prithvi neural network. And earchaccess to connect to NASA Earthdata and search HLS satellite data.

After our imports we make our folder settings and make it so our outputs all go into data and then to their own folders as well. We make local variables to help organize where we are in the process of our code, and make our 5 states for our model.

For each date window, we made it so we search up to 100 HLS granules and only open the best 12 candidate images. It rejects an image chip if more than 80% of its pixels are cloudy or bad. And requires 5% of usable pixels in order for it to use that image

We made our boundaries for each state and then mapped them in using the geojson files. We also set our model to which year we are working with. 

We found our end of season dates using AI to just give us a general date for each. Wisconson and Colorado had Oct 20, Iowa Oct. 10, Nebraska Oct. 5, and Missouri Sep. 30

Our make_temporal_window creates a search window around a forecast date. Band order is the six satellite bands used for the model chip. Red, green, blue for the picture, Nir for vegetation strength and SWIR for moisture and stress signals

It also sets environment options so GDAL can read NASA cloud-hosted satellite files. And takes satellite data and returns a normal decimal number to work with.

After reads one state’s GeoJSON boundary and loads map file to our folders.

For each state we create a small square region inside the state because the whole state is too big for Prithvi to process. So we create one model-sized chip inside the state.

For that image we get, it tries to read cloud cover from the satellite metadata and looks for cloud cover fields

We also find out if the image is Sentinel HLS or Landsat HLS and image capture time

We then do a lot with bands and eventually require six bands plus cloud mask. We need it to find the URLs for the needed band

Searches NASA for images over Region Of Interest (ROI) and date window.

In NASA’s earthdata we add found images to a list and put most cloudy images last.

Opens one satellite band from the cloud It reads directly from the cloud but does not download the whole scene.

Our cloudmask creates a mask of bad pixels. That could be clouds, cloud shadows, snow/ice, water, etc.

We also get the calculated vegetation greenness from the ndvi. Formula is (NIR - Red) / (NIR + Red)

We also use a submethod to calculate the Enhanced Vegetation Index (EVI) as well. It then saves a preview image so we could see what it did on our end in the testing stage.

The open_candidate_to_chip  tries to turn one satellite granule into a usable model image.
At the end it creates a final chip where it isn’t too cloudy and has enough good pixels and can output a row with cloud %, EVI, NDVI, and a preview path


The temporal stack combines image chips from different dates.

Example:
August 1 chip
September 1 chip
October 1 chip
End season chip
Becomes:
4 x 6 x 224 x 224

We then finally push it all to Prithvi and output the Prithvi features

Then turns Prithvi output into simple numbers.
feature_shape
feature_mean
feature_std
feature_min
Feature_max

And the main method just runs everything together

The final csv put values in Prithvi and are statistics of the Prithvi model’s internal features including:
feature_mean
feature_std
feature_min
feature_max
Basically it summarized all of them together and the model looked at the satellite images and turned them into a giant vector of numbers/features.