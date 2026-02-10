## Bonjour/Hi 👋

My name is Louis, Louis Fillon. I’m passionate about data science and art, with a background as a financial auditor, entrepreneur, and business analyst.

In this repository, you’ll find projects that showcase my skills, along with a few little goodies—because I treat it like a Pokédex of my epicurean life.

***Disclaimer***: There won’t be any personal music prereleases in the Music folder.

<!--(add track excerpts titled "preview track.wav" and a mockup of the cover/calligram)-->

Looking forward to chatting about anything!

### Project index

- **[Module for analysis, robustness assessment, and reporting based on structured and unstructured data (on public World Bank projects)](./impactlens_app/)**
   - Designed an end-to-end data pipeline (bronze/silver/gold):
     - Ingested data from World Bank APIs (projects + WDS)
     - Normalized and typed datasets
     - Selected relevant documents
     - Ensured traceability via manifests and dataset_version_id versioning
     - Stored datasets in Parquet format (Polars/PyArrow)
   - Processed documents and performed OCR:
     - Downloaded PDFs and extracted text (PyMuPDF)
     - Used OCR fallback (EasyOCR) for FR/EN documents
     - Chunked pages with quality metrics for analysis
   - Built information extraction using an LLM (Ollama) with grounding:
     - Defined strict Pydantic schemas for outputs
     - Required mandatory citations (doc/page/quote)
     - Implemented automated anti-hallucination validation (quote-in-chunk)
     - Explicitly handled uncertainty (“insufficient evidence” / “not confirmed”)
   - Evaluated and audited model performance:
     - Used gold sets to assess accuracy
     - Developed external assessment module for grounding support rate, completeness, inter-run stability, and heuristic consistency (with optional judge)
     - Integrated CLI/API/UI (Typer, FastAPI, Streamlit) for execution, inspection, and report export


- **[X-Ray imaging analysis application integrated into the medical process (Note: 30/30)](./xray_analysis_app/)**
  - Designed a CNN-based diagnostic support system to digitize and automate the workflow from Preparer to Doctor for pneumonia analysis.
  - Managed DICOM/image data processing including ingestion, metadata extraction, exploratory data analysis, and construction of an end-to-end pipeline.
  - Implemented CNN model inference (scoring and classification) alongside a user interface for annotation and validation of predictions.
  - Set up continuous improvement and full traceability of changes through versioning, audit trails, and analytical exports to support monitoring and a feedback loop.

- **[Resistance distance and criticality in supply chain networks: comparison with classic centrality measures in SCRM (Note: A)](./scrm_resistance_distance/)**
  - Defined a complex network analysis approach for supply chain risk management to support decision-making.
  - Designed criticality metrics and a robustness testing protocol (stress tests) to evaluate the impact of disruptions on the network.
  - Validated and compared rankings using statistical methods such as correlations and top-k comparisons to deliver interpretable results.
  - Ensured reproducibility through structured Python workflows covering scripts, visualization, and reporting.

- **[Sentiment analysis with NLP on Amazon Reviews data (Note: 30/30)](./sentimentanalysis_amazon/)**
  - Developed a pipeline for multilingual text analysis including preprocessing, translation, and standardization.
  - Prepared data and labeling procedures and established a reproducible evaluation framework.
  - Compared machine learning approaches with deep learning/transformer-based methods and conducted error and bias analyses to explain outcomes.
  - Delivered reports, visualizations, and exports to enable iteration, tracking, and clear communication of recommendations.

- **[Kaggle competitions and other class projects](./inclass_kaggle_competitions/)**
  - Achieved 1st place in a customer retention project using Random Forest.
  - Earned 3rd place in a mobile game data mining project using Neural Networks and Gradient Boosting.
  - Placed 4th in a marketing catalog campaign project using Logistic Regression.
  - Completed a segmentation and marketing strategy project in a Big Data context for a large enterprise under a non-disclosure agreement.