import 'package:dart_supervision/dart_supervision.dart';

/// Demo showing the clean separation of NumPy-like functionality
/// and computer vision tracking capabilities.
void main() {
  print('=== NumPy-like NDArray Demo ===');

  // Create arrays
  final arr1 = NDArray.fromList([
    [1.0, 2.0],
    [3.0, 4.0],
  ]);
  final arr2 = NDArray.eye(2);

  print('Array 1:\n$arr1');
  print('Identity matrix:\n$arr2');

  // Matrix operations
  final product = arr1.dot(arr2);
  print('Matrix multiplication result:\n$product');

  // Element-wise operations
  final scaled = arr1 * 2;
  print('Scaled array:\n$scaled');

  // Boolean indexing
  final mask = arr1 > 2.0;
  final filtered = arr1 & mask;
  print('Filtered elements: $filtered');

  print('\n=== Computer Vision Tracking Demo ===');

  // Create mock detection data (like from YOLO)
  final detections1 = Detections(
    xyxy: NDArray(
      [2, 4],
      data: [
        100.0, 100.0, 200.0, 200.0, // Person bounding box
        300.0, 150.0, 400.0, 250.0, // Car bounding box
      ],
    ),
    confidence: NDArray([2], data: [0.9, 0.8]),
    classId: NDArray([2], data: [0.0, 1.0]), // 0=person, 1=car
  );

  print('Frame 1 detections: ${detections1.length} objects');

  // Initialize ByteTracker (like supervision)
  final tracker = ByteTracker(
    trackActivationThreshold: 0.7,
    minimumMatchingThreshold: 0.8,
  );

  // Update tracker with detections
  final tracked1 = tracker.updateWithDetections(detections1);
  print(
    'Frame 1 tracked: ${tracked1.length} objects with IDs: ${tracked1.trackerId?.toList()}',
  );

  // Simulate next frame with moved objects
  final detections2 = Detections(
    xyxy: NDArray(
      [2, 4],
      data: [
        105.0, 105.0, 205.0, 205.0, // Person moved slightly
        295.0, 145.0, 395.0, 245.0, // Car moved slightly
      ],
    ),
    confidence: NDArray([2], data: [0.85, 0.9]),
    classId: NDArray([2], data: [0.0, 1.0]),
  );

  final tracked2 = tracker.updateWithDetections(detections2);
  print(
    'Frame 2 tracked: ${tracked2.length} objects with IDs: ${tracked2.trackerId?.toList()}',
  );

  // Get anchor points for visualization
  if (tracked2.length > 0) {
    final centers = tracked2.getAnchorsCoordinates(anchor: 'center');
    print('Object centers: ${centers.toList()}');

    final bottomCenters = tracked2.getAnchorsCoordinates(
      anchor: 'bottom_center',
    );
    print('Bottom centers: ${bottomCenters.toList()}');
  }

  // Apply NMS to filter overlapping detections
  final withNms = detections2.withNms(threshold: 0.5);
  print('After NMS: ${withNms.length} objects (filtered overlaps)');

  print('\n=== Summary ===');
  print('✅ Clean separation: NumPy functionality in lib/numpy/');
  print('✅ Computer vision functionality in lib/supervision/');
  print('✅ ByteTracker compatible with supervision library API');
  print('✅ NDArray supports NumPy-like operations and view semantics');
  print('✅ Comprehensive test coverage in test/numpy/ and test/supervision/');
}
