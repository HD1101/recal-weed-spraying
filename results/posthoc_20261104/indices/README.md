# Calibration/test indices

Catalog CSVs contain only public image IDs/filenames, dimensions and group IDs.
All integer image indices are zero based and refer to the bound catalog order.
`formal/` preserves the existing formal index lists byte-for-byte. They are shared
across detector/training seeds and nozzle counts; K-prime is an aggregate count,
not a spatial annotation. `D006/` contains the same authorized nested resampling,
reconstructed from its fixed rule without computing any predictions or risks.
Its calibration/pool/test hashes match all 81,000 stored D006 index-audit rows.
The formal lists match 10,000 stored main-CRC draws (one nozzle; five datasets).
No image, mask, bounding-box truth label or private provenance file is included.
The original/private verification and lineage records remain with the authors.
