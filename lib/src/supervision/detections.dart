import '../numpy/ndarray.dart';

/// Represents a collection of object detections in an image.
///
/// This class is similar to the supervision library's Detections class,
/// storing bounding boxes, confidence scores, class IDs, and optional tracker IDs.
class Detections {
  /// Bounding boxes in xyxy format (x1, y1, x2, y2)
  final NDArray xyxy;

  /// Confidence scores for each detection
  final NDArray? confidence;

  /// Class IDs for each detection
  final NDArray? classId;

  /// Tracker IDs for each detection (assigned by tracker)
  final NDArray? trackerId;

  Detections({
    required this.xyxy,
    this.confidence,
    this.classId,
    this.trackerId,
  }) {
    if (xyxy.ndim != 2 || xyxy.shape[1] != 4) {
      throw ArgumentError('xyxy must be a 2D array with shape [N, 4]');
    }

    final n = xyxy.shape[0];

    if (confidence != null &&
        (confidence!.ndim != 1 || confidence!.shape[0] != n)) {
      throw ArgumentError('confidence must be a 1D array with length $n');
    }

    if (classId != null && (classId!.ndim != 1 || classId!.shape[0] != n)) {
      throw ArgumentError('classId must be a 1D array with length $n');
    }

    if (trackerId != null &&
        (trackerId!.ndim != 1 || trackerId!.shape[0] != n)) {
      throw ArgumentError('trackerId must be a 1D array with length $n');
    }
  }

  /// Creates an empty Detections object.
  factory Detections.empty() {
    return Detections(xyxy: NDArray([0, 4]));
  }

  /// Returns the number of detections.
  int get length => xyxy.shape[0];

  /// Returns true if there are no detections.
  bool get isEmpty => length == 0;

  /// Returns anchor coordinates for each detection.
  /// Position can be 'center', 'top_left', 'bottom_center', etc.
  NDArray getAnchorsCoordinates({String anchor = 'center'}) {
    if (isEmpty) return NDArray([0, 2]);

    final result = NDArray([length, 2]);

    for (int i = 0; i < length; i++) {
      final x1 = xyxy[[i, 0]];
      final y1 = xyxy[[i, 1]];
      final x2 = xyxy[[i, 2]];
      final y2 = xyxy[[i, 3]];

      double x, y;

      switch (anchor) {
        case 'center':
          x = (x1 + x2) / 2;
          y = (y1 + y2) / 2;
          break;
        case 'top_left':
          x = x1;
          y = y1;
          break;
        case 'bottom_center':
          x = (x1 + x2) / 2;
          y = y2;
          break;
        default:
          throw ArgumentError('Unsupported anchor: $anchor');
      }

      result[[i, 0]] = x;
      result[[i, 1]] = y;
    }

    return result;
  }

  /// Filters detections based on indices.
  Detections operator [](NDArray indices) {
    if (indices.ndim != 1) {
      throw ArgumentError('Indices must be a 1D array');
    }

    final selectedCount = indices.size;
    final newXyxy = NDArray([selectedCount, 4]);

    for (int i = 0; i < selectedCount; i++) {
      final idx = indices[[i]].round();
      if (idx < 0 || idx >= length) {
        throw RangeError('Index $idx is out of bounds for length $length');
      }

      for (int j = 0; j < 4; j++) {
        newXyxy[[i, j]] = xyxy[[idx, j]];
      }
    }

    NDArray? newConfidence;
    if (confidence != null) {
      newConfidence = NDArray([selectedCount]);
      for (int i = 0; i < selectedCount; i++) {
        final idx = indices[[i]].round();
        newConfidence[[i]] = confidence![[idx]];
      }
    }

    NDArray? newClassId;
    if (classId != null) {
      newClassId = NDArray([selectedCount]);
      for (int i = 0; i < selectedCount; i++) {
        final idx = indices[[i]].round();
        newClassId[[i]] = classId![[idx]];
      }
    }

    NDArray? newTrackerId;
    if (trackerId != null) {
      newTrackerId = NDArray([selectedCount]);
      for (int i = 0; i < selectedCount; i++) {
        final idx = indices[[i]].round();
        newTrackerId[[i]] = trackerId![[idx]];
      }
    }

    return Detections(
      xyxy: newXyxy,
      confidence: newConfidence,
      classId: newClassId,
      trackerId: newTrackerId,
    );
  }

  /// Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
  Detections withNms({double threshold = 0.5}) {
    if (isEmpty || confidence == null) return this;

    // Sort detections by confidence (descending)
    final indices = List.generate(length, (i) => i);
    indices.sort((a, b) => confidence![[b]].compareTo(confidence![[a]]));

    final keep = <int>[];
    final suppressed = List.filled(length, false);

    for (final i in indices) {
      if (suppressed[i]) continue;

      keep.add(i);

      // Suppress overlapping detections
      for (final j in indices) {
        if (i == j || suppressed[j]) continue;

        final iou = _computeIoU(i, j);
        if (iou > threshold) {
          suppressed[j] = true;
        }
      }
    }

    // Create filtered detections
    final keepIndices = NDArray([
      keep.length,
    ], data: keep.map((i) => i.toDouble()).toList());
    return this[keepIndices];
  }

  /// Computes IoU between two detections.
  double _computeIoU(int i, int j) {
    final x1_i = xyxy[[i, 0]];
    final y1_i = xyxy[[i, 1]];
    final x2_i = xyxy[[i, 2]];
    final y2_i = xyxy[[i, 3]];

    final x1_j = xyxy[[j, 0]];
    final y1_j = xyxy[[j, 1]];
    final x2_j = xyxy[[j, 2]];
    final y2_j = xyxy[[j, 3]];

    // Compute intersection
    final x1 = x1_i > x1_j ? x1_i : x1_j;
    final y1 = y1_i > y1_j ? y1_i : y1_j;
    final x2 = x2_i < x2_j ? x2_i : x2_j;
    final y2 = y2_i < y2_j ? y2_i : y2_j;

    if (x2 <= x1 || y2 <= y1) return 0.0;

    final intersection = (x2 - x1) * (y2 - y1);

    // Compute union
    final area_i = (x2_i - x1_i) * (y2_i - y1_i);
    final area_j = (x2_j - x1_j) * (y2_j - y1_j);
    final union = area_i + area_j - intersection;

    return intersection / union;
  }

  @override
  String toString() {
    return 'Detections(count: $length, '
        'has_confidence: ${confidence != null}, '
        'has_class_id: ${classId != null}, '
        'has_tracker_id: ${trackerId != null})';
  }
}
