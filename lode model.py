import gzip
import base64

MODEL_FILE = "models/url_model_15trees.pkl"
OUTPUT_FILE = "embedded_url_model.txt"

print("Reading trained URL model...")

with open(MODEL_FILE, "rb") as f:
    model_bytes = f.read()

print(f"Original size: {len(model_bytes) / (1024 * 1024):.2f} MB")

print("Compressing model...")

compressed = gzip.compress(
    model_bytes,
    compresslevel=9
)

print(
    f"Compressed size: "
    f"{len(compressed) / (1024 * 1024):.2f} MB"
)

encoded = base64.b64encode(
    compressed
).decode("ascii")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(encoded)

print("Done!")
print("Created:", OUTPUT_FILE)