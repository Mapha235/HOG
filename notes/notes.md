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

**Presentation ideas**
- describe the goal of HOG
- give a general idea of useful features 
- define a gradient, gradient image
- introduce the feature extraction and object detection chain
- 