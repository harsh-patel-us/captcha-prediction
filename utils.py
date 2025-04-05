def predict_captcha(image, model):
    """Predicts the text from a captcha image using a YOLO model."""
    results = model(image)  # Pass the PIL image directly

    predictions = []
    for result in results:
        for box in result.boxes:
            class_id = box.cls.item()
            character = model.names[int(class_id)]
            predictions.append(
                (box.xyxy[0][0].item(), character)
            )  # (x-coordinate, character)

    captcha_text = "".join([char for _, char in sorted(predictions)])

    return captcha_text
