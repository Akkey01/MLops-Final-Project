import logging
from pathlib import Path
from optimum.exporters.onnx import main_export
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="🟢 [%(levelname)s] %(message)s"
)

# Define paths
MODEL_DIR = Path("./tinyllama11b_chat_ft1")
ONNX_DIR = MODEL_DIR / "onnx"
TASK = "text-generation"

logging.info("Starting ONNX export for model at: %s", MODEL_DIR)

if not MODEL_DIR.exists():
    logging.error("Model path %s does not exist!", MODEL_DIR)
    sys.exit(1)

try:
    main_export(
        model_name_or_path=str(MODEL_DIR.resolve()),
        output=str(ONNX_DIR.resolve()),
        task=TASK,
        opset=17,
        framework="pt",
        library="transformers"  # required for local paths
    )

    logging.info("✅ ONNX export completed successfully!")
    logging.info("Exported files saved to: %s", ONNX_DIR.resolve())

except Exception as e:
    logging.error("❌ ONNX export failed: %s", str(e))
    sys.exit(1)