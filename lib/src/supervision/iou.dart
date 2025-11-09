import '../numpy/ndarray.dart';

/// Computes Intersection over Union (IoU) between bounding boxes.
///
/// Supports both single box pairs and batch operations.
class IoUCalculator {
  /// Computes IoU between two bounding boxes in xyxy format.
  static double iou(NDArray box1, NDArray box2) {
    if (box1.size != 4 || box2.size != 4) {
      throw ArgumentError('Boxes must have 4 coordinates each');
    }

    // Extract coordinates
    final x1_1 = box1[[0]];
    final y1_1 = box1[[1]];
    final x2_1 = box1[[2]];
    final y2_1 = box1[[3]];

    final x1_2 = box2[[0]];
    final y1_2 = box2[[1]];
    final x2_2 = box2[[2]];
    final y2_2 = box2[[3]];

    // Compute intersection
    final xLeft = x1_1 > x1_2 ? x1_1 : x1_2;
    final yTop = y1_1 > y1_2 ? y1_1 : y1_2;
    final xRight = x2_1 < x2_2 ? x2_1 : x2_2;
    final yBottom = y2_1 < y2_2 ? y2_1 : y2_2;

    if (xRight <= xLeft || yBottom <= yTop) {
      return 0.0;
    }

    final intersection = (xRight - xLeft) * (yBottom - yTop);

    // Compute union
    final area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    final area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    final union = area1 + area2 - intersection;

    return intersection / union;
  }

  /// Computes IoU matrix between two sets of bounding boxes.
  static NDArray iouBatch(NDArray boxes1, NDArray boxes2) {
    if (boxes1.ndim != 2 || boxes1.shape[1] != 4) {
      throw ArgumentError('boxes1 must be [N, 4] array');
    }
    if (boxes2.ndim != 2 || boxes2.shape[1] != 4) {
      throw ArgumentError('boxes2 must be [M, 4] array');
    }

    final n = boxes1.shape[0];
    final m = boxes2.shape[0];
    final result = NDArray([n, m]);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        final box1 = NDArray(
          [4],
          data: [
            boxes1[[i, 0]],
            boxes1[[i, 1]],
            boxes1[[i, 2]],
            boxes1[[i, 3]],
          ],
        );
        final box2 = NDArray(
          [4],
          data: [
            boxes2[[j, 0]],
            boxes2[[j, 1]],
            boxes2[[j, 2]],
            boxes2[[j, 3]],
          ],
        );
        result[[i, j]] = iou(box1, box2);
      }
    }

    return result;
  }

  /// Computes IoU distances (1 - IoU) for tracking.
  static NDArray iouDistance(NDArray boxes1, NDArray boxes2) {
    final ious = iouBatch(boxes1, boxes2);
    return NDArray.ones(ious.shape) - ious;
  }
}
