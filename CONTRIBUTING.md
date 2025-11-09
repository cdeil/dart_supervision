## What is this?

Python has a nice ecosystem of scientific computing and computer vision packages.
I was using [Numpy](https://numpy.org), [Scipy](https://scipy.org), [OpenCV](https://opencv.org),
[Ultralytics](https://www.ultralytics.com) and [RFDetr](https://rfdetr.roboflow.com) and several
other tools to prototype my computer vision application.

The [Supervision](https://supervision.roboflow.com) package has a nice set of simple tools
to work with images, videos, detections, and added a few essential algorithms especially
the [ByteTrack object tracker](https://supervision.roboflow.com/latest/how_to/track_objects).

When I tried to bring the prototype to Dart and Flutter for mobile iOS and Android app deployment
I found a much more fragmented and immature ecosystem of packagesa and specifically I couldn't find
`ByteTrack`, and this is how `dart_supervision` was born.

## Feedback and help needed!

This is a very new and immature package and it's core design isn't figured out yet.
I just put it on Github and pub.dev because I wanted to learn how to make a Dart package
and maybe get some feedback.

I didn't find a good numerical array (Numpy) and scientific computing (Scipy) equivalent in Dart.
So I had a chat with Gemini and then decided to simply let the coding agent write a `NDarray` and replicate the required Scipy methods
for the `ByteTrack` implementation from scratch. These are questionable choices of course.
Specifically I'd be interested if [typed_data](https://pub.dev/packages/typed_data) or
[vector_math](https://pub.dev/packages/vector_math) or [ml_linalg](https://pub.dev/packages/ml_linalg)
or some other package(s) would give a better foundation.

For the image side I was shocked to learn that the [dart:ui Image class](https://api.flutter.dev/flutter/dart-ui/Image-class.html)
has no support to read a JPEG image and the [Flutter image class](https://api.flutter.dev/flutter/widgets/Image-class.html)
is a widget that is not easily usable e.g. in Dart tests. For videos the [video_player](https://pub.dev/packages/video_player)
does not support accessing the image frames for object detection or drawing or computer vision analysis,
and again it's a Flutter plugin that can't be used in Dart tests for a computer vision library.
The Python Supervision package uses [OpenCV](https://opencv.org) as a core dependency for many functions
(image and video I/O, drawing, polygon/mask conversions, ...). I see there is [dartcv4](https://pub.dev/packages/dartcv4)
and [opencv_core](https://pub.dev/packages/opencv_core) and [opencv_dart](https://pub.dev/packages/opencv_dart)
for Dart and Flutter as well as [mediakit](https://pub.dev/packages/media_kit) and some other image/video packages.
But those are big dependencies which I didn't want to put in our app. So long story short, for now
I chose the pure Dart [image](https://pub.dev/packages/image) package to work with images. It doesn't support video
so for now we only support animated GIF and WebP for video. Ideally I would like to not have any required
dependency on these image and video packages, but have a few utilities to read and convert between these formats
and then any user of this `dart_supervision` package could choose whatever they like. Similar to how the
Supervision Python package supports OpenCV and [Pillow](https://pillow.readthedocs.io) and the user can choose.
If anyone has feedback which dependencies to choose or how to handle this situation nicely please open a Github issue!

For now I implemeted a few of the supervision annotators to draw detections on an image or video frame
wrapping the [image](https://pub.dev/packages/image) package functions (as opposed to OpenCV in the Python supervision).
Here I'm thinking maybe instead we could move drawing to a `flutter_supervision` package and there use the native
Flutter drawing functionality and possibly even open up interactivity for annotating?
This could allow us to remove the `image` dependency in `dart_supervision` and have a zero-dependency pure Dart package
which basically then offers the `ByteTrack` and `Detections` classes, not much more.

Lastly the situation with object detectors is similar. Supervision has the `Detections` class with
`from_ultralytics` and `from_sam` and a dozen other object detection Python packages, but doesn't have
an installation dependency on any of them. If anyone knows how to achieve the same in Dart let me know please!
Concretely I think [ultralytics_yolo](https://pub.dev/packages/ultralytics_yolo) and
[google_mlkit_object_detection](https://pub.dev/packages/google_mlkit_object_detection) would be good to support
somehow? Any other object detection or tracking packages to try or collaborate with?
Or maybe instead of trying to interface with packages instead support common computer vision model load and predict
directly somehow?
