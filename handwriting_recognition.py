
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

MODEL_NAME = "microsoft/trocr-small-handwritten"

# Load model once
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)


def recognize_handwriting(image):
    """
    Recognize English handwriting from an image.
    Accepts a PIL Image and returns recognized text.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")

    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


def recognize_page(image):
    """
    Recognize all handwritten lines from a complete page.
    Returns the extracted text with line breaks.
    """

    from src.segmentation import segment_lines

    lines = segment_lines(image)

    if not lines:
        return "No handwritten text detected."

    recognized_lines = []

    for i, line in enumerate(lines):
        line_image = Image.fromarray(line)

        text = recognize_handwriting(line_image)

        recognized_lines.append(text)

        print(f"Line {i + 1}: {text}")

    return "\n".join(recognized_lines)
