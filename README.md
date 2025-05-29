# CE_LDR (Implementation of "Contrast Enhancement Based on Layered Difference Representation of 2D Histograms")

This Python script implements the image contrast enhancement algorithm described in the paper "Contrast Enhancement Based on Layered Difference Representation of 2D Histograms."

---

## Algorithm

The core logic of this script is based on the method detailed in the scientific publication: "Contrast Enhancement Based on Layered Difference Representation of 2D Histograms." For a comprehensive understanding of the underlying principles, mathematical formulations, and evaluation, please refer to the original paper.

---

## Dependencies

To run `CE_LDR.py`, you need the following Python libraries:

*   **NumPy**: For numerical operations, especially array manipulations.
*   **Pillow (PIL)**: For image file loading and handling.
*   **Matplotlib**: For displaying images.

You can install these dependencies using pip:

```bash
pip install numpy Pillow matplotlib
```

---

## Usage

**a. Navigate to the project directory:**

```shell
cd /path/to/CE_LDR/
```
*(Replace `/path/to/CE_LDR/` with the actual path to the directory containing `CE_LDR.py`)*

**b. Execute the script:**

The script is run from the command line with the following syntax:

```shell
python CE_LDR.py --input_path ${INPUT_PATH} [--alpha ${ALPHA}] [--U ${U_PATH}]
```

---

## Parameters

*   `--input_path ${INPUT_PATH}` (required):
    *   Specifies the path to the input image file that you want to enhance.
    *   Example: `images/my_photo.jpg`, `../data/landscape.png`.

*   `--alpha ${ALPHA}` (optional):
    *   A floating-point number that controls the intensity of the contrast enhancement.
    *   Higher values generally lead to stronger enhancement.
    *   If not specified, this parameter defaults to `2.5`.
    *   Example: `--alpha 3.0`.

*   `--U ${U_PATH}` (optional):
    *   Specifies the path to a `.npy` file that contains the U matrix used in equation 31 of the paper. This matrix is part of the Layered Difference Representation.
    *   The matrix stored in the `.npy` file should have a shape of (255, 255).
    *   If this parameter is not provided, the script will calculate a default U matrix based on the algorithm's formulas.
    *   Example: `--U my_custom_U_matrix.npy`.

---

## Output

When `CE_LDR.py` is executed:

1.  **Image Display**:
    *   The original input image will be displayed in a Matplotlib window.
    *   After processing, the contrast-enhanced image will be displayed in a new Matplotlib window.

2.  **Saved Image**:
    *   The enhanced image is saved as a PNG file in the same directory where the `CE_LDR.py` script is located.
    *   The output filename is generated based on the input filename using the format: `ce_<original_filename_without_extension>_<original_extension>.png`.
    *   For example:
        *   If input is `test.png`, output will be `ce_test_png.png`.
        *   If input is `../images/sample.jpg`, output will be `ce_sample_jpg.png`.

---
