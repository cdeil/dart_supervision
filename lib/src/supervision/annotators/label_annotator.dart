import 'package:image/image.dart' as img;
import '../detections.dart';

/// Annotator for drawing text labels above detections.
///
/// Similar to supervision's LabelAnnotator, this class provides methods
/// to draw text labels on images with customizable appearance.
class LabelAnnotator {
  final img.ColorRgb8 textColor;
  final img.ColorRgb8 backgroundColor;
  final int fontSize;
  final int padding;
  final String fontFamily;

  /// Creates a LabelAnnotator with the specified appearance.
  ///
  /// [textColor] - Color of the text
  /// [backgroundColor] - Background color of the label
  /// [fontSize] - Size of the font
  /// [padding] - Padding around the text
  /// [fontFamily] - Font family name (currently limited by image package)
  LabelAnnotator({
    img.ColorRgb8? textColor,
    img.ColorRgb8? backgroundColor,
    this.fontSize = 14,
    this.padding = 4,
    this.fontFamily = 'arial',
  })  : textColor = textColor ?? img.ColorRgb8(255, 255, 255), // White text
        backgroundColor =
            backgroundColor ?? img.ColorRgb8(0, 0, 0); // Black background

  /// Annotates an image with labels from detections.
  ///
  /// [image] - The image to annotate
  /// [detections] - The detections to label (in pixel coordinates)
  /// [labels] - List of labels for each detection (optional)
  ///
  /// If labels is not provided, default labels will be generated from
  /// class IDs, tracker IDs, and confidence scores.
  void annotate(
    img.Image image,
    Detections detections, {
    List<String>? labels,
  }) {
    if (detections.isEmpty) return;

    final effectiveLabels = labels ?? _generateDefaultLabels(detections);

    for (int i = 0; i < detections.length; i++) {
      if (i >= effectiveLabels.length) break;

      // Get pixel coordinates and clamp to image bounds
      final x1 = detections.xyxy[[i, 0]].clamp(0, image.width - 1).toInt();
      final y1 = detections.xyxy[[i, 1]].clamp(0, image.height - 1).toInt();

      // Draw the label
      _drawLabel(image, x1, y1, effectiveLabels[i]);
    }
  }

  /// Generates default labels for detections.
  List<String> _generateDefaultLabels(Detections detections) {
    final labels = <String>[];

    for (int i = 0; i < detections.length; i++) {
      final parts = <String>[];

      // Add class ID if available
      if (detections.classId != null) {
        final classId = detections.classId![[i]].round();
        parts.add('class:$classId');
      }

      // Add tracker ID if available
      if (detections.trackerId != null) {
        final trackerId = detections.trackerId![[i]].round();
        parts.add('id:$trackerId');
      }

      // Add confidence if available
      if (detections.confidence != null) {
        final confidence = detections.confidence![[i]];
        parts.add('${(confidence * 100).toStringAsFixed(1)}%');
      }

      labels.add(parts.isEmpty ? 'detection' : parts.join(' '));
    }

    return labels;
  }

  /// Draws a single label on the image.
  void _drawLabel(img.Image image, int x, int y, String text) {
    // Ensure coordinates are within image bounds
    final clampedX = x.clamp(0, image.width - 1);
    var clampedY = y.clamp(0, image.height - 1);

    // Estimate text dimensions (rough approximation)
    final textWidth = text.length * (fontSize ~/ 2);
    final textHeight = fontSize + 4;

    // Calculate label background position
    final labelX1 = clampedX;
    final labelY1 = (clampedY - textHeight - padding)
        .clamp(0, image.height - textHeight - padding);
    final labelX2 =
        (labelX1 + textWidth + 2 * padding).clamp(0, image.width - 1);
    final labelY2 = labelY1 + textHeight + padding;

    // Draw background rectangle using latest API
    img.fillRect(
      image,
      x1: labelX1,
      y1: labelY1,
      x2: labelX2,
      y2: labelY2,
      color: backgroundColor,
    );

    // Draw text
    // Note: The image package has limited text rendering capabilities
    // For now, we'll draw a simple text representation
    _drawSimpleText(image, labelX1 + padding, labelY1 + padding, text);
  }

  /// Draws simple text using basic character shapes.
  /// This is a fallback since the image package has limited font support.
  void _drawSimpleText(img.Image image, int x, int y, String text) {
    // For now, just draw small rectangles as a text placeholder
    // In a real implementation, you might want to use a proper font rendering library

    final charWidth = fontSize ~/ 2;
    final charHeight = fontSize;

    for (int i = 0; i < text.length; i++) {
      final charX = x + i * charWidth;
      final charY = y;

      if (charX + charWidth <= image.width &&
          charY + charHeight <= image.height) {
        // Draw a simple character representation using latest API
        img.fillRect(
          image,
          x1: charX,
          y1: charY,
          x2: charX + charWidth - 1,
          y2: charY + charHeight - 1,
          color: textColor,
        );
      }
    }
  }


}
