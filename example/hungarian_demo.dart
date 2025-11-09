import 'package:dart_supervision/dart_supervision.dart';

void main() {
  print('=== Linear Sum Assignment (Hungarian Algorithm) Demo ===\n');

  // Example 1: Basic assignment problem
  print('Example 1: Worker-Job Assignment Problem');
  print('Cost matrix (workers × jobs):');
  final costMatrix = NDArray.fromList([
    [4.0, 1.0, 3.0], // Worker 0 costs
    [2.0, 0.0, 5.0], // Worker 1 costs
    [3.0, 2.0, 2.0], // Worker 2 costs
  ]);

  for (int i = 0; i < 3; i++) {
    print(
      'Worker $i: [${costMatrix[[i, 0]]}, ${costMatrix[[i, 1]]}, ${costMatrix[[
        i,
        2
      ]]}]',
    );
  }

  final result = LinearSumAssignment.solve(costMatrix);
  print('\nOptimal Assignment (minimize cost):');
  for (int i = 0; i < result.rowIndices.length; i++) {
    final worker = result.rowIndices[i];
    final job = result.colIndices[i];
    final cost = costMatrix[[worker, job]];
    print('Worker $worker → Job $job (cost: $cost)');
  }
  print('Total minimum cost: ${result.totalCost(costMatrix)}\n');

  // Example 2: Maximization problem
  print('Example 2: Profit Maximization');
  final resultMax = LinearSumAssignment.solve(costMatrix, maximize: true);
  print('Optimal Assignment (maximize profit):');
  for (int i = 0; i < resultMax.rowIndices.length; i++) {
    final worker = resultMax.rowIndices[i];
    final job = resultMax.colIndices[i];
    final profit = costMatrix[[worker, job]];
    print('Worker $worker → Job $job (profit: $profit)');
  }
  print('Total maximum profit: ${resultMax.totalCost(costMatrix)}\n');

  // Example 3: Rectangular matrix (more workers than jobs)
  print('Example 3: More Workers than Jobs');
  final rectMatrix = NDArray.fromList([
    [5.0, 3.0], // Worker 0
    [1.0, 2.0], // Worker 1
    [4.0, 6.0], // Worker 2
    [2.0, 1.0], // Worker 3
  ]);

  print('Cost matrix (4 workers × 2 jobs):');
  for (int i = 0; i < 4; i++) {
    print('Worker $i: [${rectMatrix[[i, 0]]}, ${rectMatrix[[i, 1]]}]');
  }

  final rectResult = LinearSumAssignment.solve(rectMatrix);
  print('\nOptimal Assignment:');
  for (int i = 0; i < rectResult.rowIndices.length; i++) {
    final worker = rectResult.rowIndices[i];
    final job = rectResult.colIndices[i];
    final cost = rectMatrix[[worker, job]];
    print('Worker $worker → Job $job (cost: $cost)');
  }
  print('Total cost: ${rectResult.totalCost(rectMatrix)}');

  // Show unassigned workers
  final assignedWorkers = Set.from(rectResult.rowIndices);
  final unassignedWorkers = <int>[];
  for (int i = 0; i < 4; i++) {
    if (!assignedWorkers.contains(i)) {
      unassignedWorkers.add(i);
    }
  }
  if (unassignedWorkers.isNotEmpty) {
    print('Unassigned workers: $unassignedWorkers\n');
  }

  // Example 4: Use in object tracking
  print('Example 4: Object Tracking Assignment');
  print('Simulating assignment of detections to tracks...');

  // Create mock detections and tracker
  final tracker = ByteTracker();

  // Frame 1: Initial detections
  final detections1 = Detections(
    xyxy: NDArray.fromList([
      [10.0, 10.0, 50.0, 50.0], // Detection 0
      [100.0, 100.0, 140.0, 140.0], // Detection 1
    ]),
    confidence: NDArray([2], data: [0.9, 0.8]),
  );

  final result1 = tracker.updateWithDetections(detections1);
  print('Frame 1: ${result1.length} tracks initialized');

  // Frame 2: Updated detections (simulating movement)
  final detections2 = Detections(
    xyxy: NDArray.fromList([
      [12.0, 12.0, 52.0, 52.0], // Moved detection (should match track 0)
      [102.0, 98.0, 142.0, 138.0], // Moved detection (should match track 1)
      [200.0, 200.0, 240.0, 240.0], // New detection
    ]),
    confidence: NDArray([3], data: [0.85, 0.9, 0.7]),
  );

  final result2 = tracker.updateWithDetections(detections2);
  print('Frame 2: ${result2.length} tracks active');

  if (result2.trackerId != null) {
    print('Track assignments:');
    for (int i = 0; i < result2.length; i++) {
      final trackId = result2.trackerId![[i]].toInt();
      print('Detection $i → Track $trackId');
    }
  }

  print('\n=== Demo Complete ===');
  print('The Hungarian algorithm ensures optimal assignment in O(n³) time,');
  print('much more efficient than the previous greedy O(n⁴) approach!');
}
