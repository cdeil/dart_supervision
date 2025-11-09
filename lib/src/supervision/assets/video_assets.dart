/// Video assets enumeration and metadata
///
/// Dart implementation of supervision.assets.list module

const String baseVideoUrl =
    "https://media.roboflow.com/supervision/video-examples/";

/// Each member of this enum represents a video asset. The value associated with each
/// member is the filename of the video.
///
/// | Enum Member            | Video Filename             | Video URL                                                                             |
/// |------------------------|----------------------------|---------------------------------------------------------------------------------------|
/// | `vehicles`             | `vehicles.mp4`             | [Link](https://media.roboflow.com/supervision/video-examples/vehicles.mp4)            |
/// | `milkBottlingPlant`    | `milk-bottling-plant.mp4`  | [Link](https://media.roboflow.com/supervision/video-examples/milk-bottling-plant.mp4) |
/// | `vehicles2`            | `vehicles-2.mp4`           | [Link](https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4)          |
/// | `groceryStore`         | `grocery-store.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/grocery-store.mp4)       |
/// | `subway`               | `subway.mp4`               | [Link](https://media.roboflow.com/supervision/video-examples/subway.mp4)              |
/// | `marketSquare`         | `market-square.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/market-square.mp4)       |
/// | `peopleWalking`        | `people-walking.mp4`       | [Link](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)      |
/// | `beach`                | `beach-1.mp4`              | [Link](https://media.roboflow.com/supervision/video-examples/beach-1.mp4)             |
/// | `basketball`           | `basketball-1.mp4`         | [Link](https://media.roboflow.com/supervision/video-examples/basketball-1.mp4)        |
/// | `skiing`               | `skiing.mp4`               | [Link](https://media.roboflow.com/supervision/video-examples/skiing.mp4)              |
enum VideoAssets {
  vehicles('vehicles.mp4'),
  milkBottlingPlant('milk-bottling-plant.mp4'),
  vehicles2('vehicles-2.mp4'),
  groceryStore('grocery-store.mp4'),
  subway('subway.mp4'),
  marketSquare('market-square.mp4'),
  peopleWalking('people-walking.mp4'),
  beach('beach-1.mp4'),
  basketball('basketball-1.mp4'),
  skiing('skiing.mp4');

  const VideoAssets(this.filename);

  /// The filename of the video asset
  final String filename;

  /// Get the full URL for this video asset
  String get url => '$baseVideoUrl$filename';

  /// Get a list of all video asset filenames
  static List<String> get allFilenames =>
      VideoAssets.values.map((e) => e.filename).toList();
}

/// Asset metadata containing URL and MD5 hash
class AssetMetadata {
  const AssetMetadata(this.url, this.md5Hash);

  final String url;
  final String md5Hash;
}

/// Video assets metadata mapping filename to URL and MD5 hash
const Map<String, AssetMetadata> videoAssetsMetadata = {
  'vehicles.mp4': AssetMetadata(
    '${baseVideoUrl}vehicles.mp4',
    '8155ff4e4de08cfa25f39de96483f918',
  ),
  'vehicles-2.mp4': AssetMetadata(
    '${baseVideoUrl}vehicles-2.mp4',
    '830af6fba21ffbf14867a7fea595937b',
  ),
  'milk-bottling-plant.mp4': AssetMetadata(
    '${baseVideoUrl}milk-bottling-plant.mp4',
    '9e8fb6e883f842a38b3d34267290bdc7',
  ),
  'grocery-store.mp4': AssetMetadata(
    '${baseVideoUrl}grocery-store.mp4',
    '11402e7b861c1980527d3d74cbe3b366',
  ),
  'subway.mp4': AssetMetadata(
    '${baseVideoUrl}subway.mp4',
    '453475750691fb23c56a0cffef089194',
  ),
  'market-square.mp4': AssetMetadata(
    '${baseVideoUrl}market-square.mp4',
    '859179bf4a21f80a8baabfdb2ed716dc',
  ),
  'people-walking.mp4': AssetMetadata(
    '${baseVideoUrl}people-walking.mp4',
    '0574c053c8686c3f1dc0aa3743e45cb9',
  ),
  'beach-1.mp4': AssetMetadata(
    '${baseVideoUrl}beach-1.mp4',
    '4175d42fec4d450ed081523fd39e0cf8',
  ),
  'basketball-1.mp4': AssetMetadata(
    '${baseVideoUrl}basketball-1.mp4',
    '60d94a3c7c47d16f09d342b088012ecc',
  ),
  'skiing.mp4': AssetMetadata(
    '${baseVideoUrl}skiing.mp4',
    'd30987cbab1bbc5934199cdd1b293119',
  ),
};
