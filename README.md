# Face Detector

Simple Python wrapper for Google's [Mediapipe Face Detection](https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md) pipeline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

These are required to use the FaceDetector module

```
numpy
opencv
tensorflow lite
```

### Anchors

To get the SSD anchors I borrowed the following script [here](https://gist.github.com/wolterlw/6f1ebc49230506f8e9ce5facc5251d4f), which is a C++ program that executes the `SsdAnchorsCalculator::GenerateAnchors` function from [this calculator](https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc).
As there's no reason to modify provided anchors I do not include it into the repository.

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.
