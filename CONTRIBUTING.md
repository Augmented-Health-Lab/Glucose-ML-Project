# Contributing Data to the Glucose-ML Collection

## Overview

Glucose-ML is a curated collection of standardized continuous glucose monitoring (CGM) datasets designed to facilitate reproducible research and machine learning applications.

While Glucose-ML is not an open-contribution repository for direct code or data integration, we actively welcome **submission of new CGM datasets** for consideration.

This document describes how you can submit dataset requests and how these datasets are integrated into Glucose-ML.

---

## How to Submit a Dataset Request

Researchers interested in contributing a CGM dataset can either:

* Open an issue on this repository describing the dataset
* Contact the Glucose-ML team via email.

When submitting a request, please include:
* A link to the dataset OR dataset request page
* A brief description of the dataset & why it should be considered.
* Information about data access (open vs. controlled)
* Any associated publications or documentation
* Licensing or reuse terms

---

## Required Dataset Components

Strong dataset candidates should contain:

### 1. Raw CGM Data

* Continuous glucose measurements per participant
* Timestamps associated with each glucose reading

### 2. Metadata

* Participant-level or dataset-level information (if available), such as:
  * Diabetes status or cohort (e.g., T1D, T2D, prediabetes, non-diabetic)
  * CGM Device
  * Sampling frequency
  * Study information

### 3. Documentation (if available)

* Description of how the data were collected
* Any preprocessing already performed
* Known limitations or data quality considerations

### *Note:* If accepted, the user is NOT responsible for standardizing the raw data. The Glucose-ML team follows a strict pipeline to consistently process and integrate newly accepted datasets. 
---

## Review and Curation Process

All submitted datasets undergo a review process prior to inclusion. Evaluation criteria include:

* Data completeness and structure
* Compatibility with Glucose-ML standardization techniques
* Clarity of documentation and metadata
* Licensing and data-sharing permissions
* Relevance to CGM research and applications

Datasets that meet these criteria will be standardized and incorporated into the Glucose-ML collection.

---

## Summary

Glucose-ML is designed to be **extensible through a structured dataset submission and harmonization process**. By standardizing diverse CGM datasets into a unified format, the platform enables scalable integration of new data while maintaining consistency and quality.

If you are interested in contributing a dataset or have questions about compatibility, please open an issue to the GitHub or reach out to the Principal Investigator: Temiloluwa Prioleau, PhD (*tpriole@emory.edu*).

---

<p>&nbsp;</p>

<p align="center">
  <img src="Logos/glucose-ml-logo_horizontal.svg" alt="Glucose-ML logo" width="450">
</p>