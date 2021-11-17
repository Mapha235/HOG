**Feature Descriptor**
  - HOG is a feature descriptor
  - representation of an image that simplifies the image by extracting useful information
  - converts an image (size x width x 3 (channels)) to a feature vector/ array of length n
    - in HOG: often 64 x 128 x 3 --> feature vector of length 3780
  - based on evaluating well-normalized local hostograms of image gradient orientation

**Questions**
- **What are useful features (for classification)?**
  - color information
  - edge information
  - in HOG: the distribution (histograms) of directions gradients (oriented gradients)
- **What information do gradients contain?**
  - magnitude of gradients
  -> x and y derivatives
  - magnitude is large around edges
  -> regions of abrupt intesity changes
- **What do gradients do?**
  - highlight outlines
  - remove non-essential information (e.g. constant colored background)
- **What is gamma?**
- **What happens when a vote falls inbetween two bins?**
  - aliasing
  - bilinear interpolation in both orientation and position
- **What is a bin?**
  - in order to construct a histogram it is necessary to "bin"/(bucket) the range of values i.e. divide the entire range of values into a series of intervals
**Presentation ideas**
- describe the goal of HOG
- give a general idea of useful features 
- define a gradient, gradient image, show the vector, how magnitude and direction are encoded
- introduce the feature extraction and object detection chain
- 

- **Links**
  - https://iq.opengenus.org/object-detection-with-histogram-of-oriented-gradients-hog/
  - https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
  - https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
  - https://learnopencv.com/histogram-of-oriented-gradients/
  - http://www.ee.unlv.edu/~b1morris/ecg782/sp14/docs/HOG_MARZIEH.pdf