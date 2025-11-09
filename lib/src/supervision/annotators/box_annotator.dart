import 'package:image/image.dart' as img;
import '../detections.dart';

/// Annotator for drawing bounding boxes around detections.
///
/// Similar to supervision's BoxAnnotator, this class provides methods
/// to draw bounding boxes on images with customizable appearance.
class BoxAnnotator {
  final img.ColorRgb8 color;
  final int thickness;
  final double opacity;

  /// Creates a BoxAnnotator with the specified appearance.
  ///
  /// [color] - Color of the bounding box lines
  /// [thickness] - Thickness of the bounding box lines
  /// [opacity] - Opacity of the bounding box (0.0 to 1.0)
  BoxAnnotator({
    img.ColorRgb8? color,
    this.thickness = 2,
    this.opacity = 1.0,
  }) : color = color ??
            img.ColorRgb8(255, 0,
                0); // Red by default  /// Annotates an image with bounding boxes from detections.
  ///
  /// [image] - The image to annotate
  /// [detections] - The detections to draw (in pixel coordinates)
  void annotate(img.Image image, Detections detections) {
    if (detections.isEmpty) return;

    // Draw each bounding box
    for (int i = 0; i < detections.length; i++) {
      // Get pixel coordinates and clamp to image bounds
      final x1 = detections.xyxy[[i, 0]].clamp(0, image.width - 1).toInt();
      final y1 = detections.xyxy[[i, 1]].clamp(0, image.height - 1).toInt();
      final x2 = detections.xyxy[[i, 2]].clamp(0, image.width - 1).toInt();
      final y2 = detections.xyxy[[i, 3]].clamp(0, image.height - 1).toInt();

      img.drawRect(
        image,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        color: color,
        thickness: thickness,
      );
    }
  }


}
