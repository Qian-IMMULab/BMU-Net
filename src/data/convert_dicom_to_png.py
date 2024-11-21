import os
import subprocess
import pydicom


def convert_dicom_to_png(dicom_path, png_path):
    os.makedirs(png_path, exist_ok=True)
    dico = pydicom.read_file(dicom_path, force=True)
    dico.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    manu = dico.Manufacturer
    png_file = os.path.join(png_path, f"{os.path.basename(dicom_path)}.png")
    if manu == "GE MEDICAL SYSTEMS":
        subprocess.call(
            ["dcmj2pnm", "+on2", "--use-voi-lut", "1", dicom_path, png_file]
        )
    elif manu == "HOLOGIC, Inc.":
        subprocess.call(
            ["dcmj2pnm", "+on2", "+Ww", "2047", "4096", dicom_path, png_file]
        )
    elif manu == "Philips Medical Systems":
        subprocess.call(
            ["dcmj2pnm", "+on2", "+Ww", "2047", "4095", dicom_path, png_file]
        )

