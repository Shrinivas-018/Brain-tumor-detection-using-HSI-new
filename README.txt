Each folder in the database contains:

- The raw original data (with 826 bands) (raw and raw.hdr files). 

- The dark and white references for calibration purposes (darkReference, whiteReference, darkReference.hdr, whiteReference.hdr files)

- The ground truth map (gtMap and gtMap.hdr files), which contain the class label matrix. As specified in the Database manuscript (https://ieeexplore.ieee.org/document/8667294) Class 0 indicates pixels that are not labelled. 1 = Normal Tissue, 2 = Tumour Tissue, 3 = Hypervascularized Tissue, 4 = Background. 

- The synthetic RGB representation of the HS cube (image.jpg) and the gtMap (gtMap.jpg).

All the files (HS image, References and gtMap) are in binary format using the ENVI format. The HDR file indicates the properties of the binary files.

To replicate the results discussed in the article "Hyperspectral Imaging Benchmark based on Machine Learning for Intraoperative Brain Tumour Detection", published in npj Precision Oncology, please use the images ranging from 004-02 to 022-03, both included.