# ClusterDF

Cluster analysis of debris flow seismic events and environmental noise signals using DEC and GMM.

---

**Author**:  Huang Zhiyong  
**Date**:    2025.04.27  
**Version**: 1.0.0

---

##  Project Description

**ClusterDF** focuses on clustering seismic signals from debris flows, rockfalls, and environmental noise using **Deep Embedded Clustering (DEC)** and **Gaussian Mixture Model (GMM)** algorithms.  
Seismic data were preprocessed using **ObsPy**, and deep learning models were implemented with **TensorFlow**.

---

##  Data Sources

- **Debris flow and environmental noise seismic data**:  
  [https://doi.org/10.12686/sed/networks/xp](https://doi.org/10.12686/sed/networks/xp)
  
- **Rockfall seismic data**:  
  [https://landslide.geologycloud.tw/map?locale=en](https://landslide.geologycloud.tw/map?locale=en)
  
- **Earthquake seismic data**:  
  [STEAD Earthquake Dataset](https://doi.org/10.1038/s41597-019-0095-6) ([Mousavi et al., 2019](https://doi.org/10.1038/s41597-019-0095-6))

---

##  Installation

To run this project, you will need to install the following Python packages:

```bash
pip install obspy==1.4.1
pip install tensorflow==2.10.0
pip install scikit-learn
pip install numpy
pip install matplotlib
