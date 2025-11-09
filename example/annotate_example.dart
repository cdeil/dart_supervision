// Example how to process "videos", i.e. animated image frames.
import 'package:image/image.dart' as img;

void drawRectangle(img.Image image) {
  img.drawRect(image,
      x1: 50,
      y1: 50,
      x2: 250,
      y2: 150,
      color: img.ColorRgb8(255, 0, 0),
      thickness: 5);
}

void main() async {
  final animation = await img.decodeImageFile('assets/people-walking.gif');
  for (final frame in animation!.frames) {
    drawRectangle(frame);
  }
  await img.encodeImageFile('assets/people-walking-annotated.gif', animation);

  final pngImage = await img.decodeImageFile('assets/people-walking.png');
  for (final frame in pngImage!.frames) {
    drawRectangle(frame);
  }
  await img.encodeImageFile('assets/people-walking-annotated.png', pngImage);

  print('Done');
}
