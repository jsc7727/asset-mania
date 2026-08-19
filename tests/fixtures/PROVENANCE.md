# Fixture provenance

Asset Mania tracks no binary fixtures. Image and Blender-header fixtures are generated inside
temporary test directories at runtime and are not copied into the repository or release
artifacts.

- Image tests create tiny synthetic 8 x 6 pixel images with Pillow.
- Blender tests write only synthetic header bytes needed by the header inspector.
- `manifest-v1-success.json` is a hand-authored UTF-8 JSON contract fixture, not an opaque
  binary asset or output captured from a user run.

If a binary fixture is added later, list its repository path, creation method or source URL,
immutable revision, SHA-256 digest, license, and redistribution evidence here before tracking it.
