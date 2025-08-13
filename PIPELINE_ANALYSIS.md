## PIPELINE REVIEW AND IMPROVEMENT PROPOSAL
### ASSESMENT OF THE CURRENT PIPELINE
#### BRIEF PIPELINE DESCRIPTION
This pipeline combines bulk (aggregated from multiple cells and cell-types) transcriptomic data with clinical data from patients to possibly:\
1. discover specific genes (or subsets of them) associated with specific subgroups of sample donors(donor stratification into biology-driven groups);\
2. discover new biomarkers associated with clinical data (tumor stage, grade, etc.) or clinical outcomes (progression free survival, overall survival, therapy response);\
It answers to the following biological questions:\
1. do different gene expression subsets exist for the same disease/cancer type?\
2. is it possible to divide the sample donors based on these molecular subtypes?\
3. is it possible to associate these subsets with clinical data from the donors?\
##### INPUT DATA
As in the first task, the input data is a molecular transcriptomic data together with clinical metadata.\
The first could be anything from sequencing files (fastq) to raw cout matrix. In any case there are bespoke pipelines that transform fastq files to count matrices (I have in mind the [nf-core rna-seq](https://nf-co.re/rnaseq/3.14.0/) which also produce extensive QC). The raw count matrix is usually organized as a matrix whose rows are genes and columns samples (associated with fastq files).\
The second one is usually a messy matrix that contains demographic data (age, gender), mutation data (if known important genes have mutations associated with tumor, Tumor Mutational Bourden, Microsatellite (In)stability, CMS), tumor-specific (histological report, stage, grade) and more cinical data (like therapy type, overall survival, progression free survival, etc.) . It often has to be consolidated by dropping redundant or zero-information descriptors.\
##### MAIN ANALYSIS STEPS
The main pipeline steps are:\
1. Data ingestion and selection of 100 most variant genes;\
2. Sample clustering based on expression data + QC;\
3. Integration of clinical data for patients stratification;\
4. Downstream analysis.\

####  BIOLOGICAL OR ANALITICAL LIMITATIONS OF THE PIPELINE
1. Expression matrix should be filtered of genes not occurring in more than e.g. 10% of the samples; \
2. Samples with less than e.g. 10% of the genes should be excluded; \
2. Expression count matrix is not batch corrected before starting the analysis; \
3. Expression count matrix is not normalized before clustering; \
4. The number of selected most variant genes is restrictive (out of >20k possible genes, 100 seem too few) and might miss biological signal; \
5. It could be useful (if sample size is large enough) to leave part of the samples out for hypotesis validation at the end of the pipeline (or even better another donor set); \
6. Integrating single cell data could be useful to pinpoint cell types associated to some particular gene expression; \


### BIOINFORMATIC ENHANCMENTS