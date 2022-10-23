library(BiocManager)
library(RTCGA)
library(RTCGA.clinical)
library(RTCGA.mRNA)

to_extract <- c(
  "patient.gender",
  "patient.age_at_initial_pathologic_diagnosis",
  "patient.race",
  "patient.ethnicity",
  "patient.clinical_cqcf.country",
  "patient.drugs.drug.therapy_types.therapy_type",
  "patient.clinical_cqcf.histological_type",
  "patient.stage_event.pathologic_stage",
  "patient.menopause_status",
  "patient.breast_carcinoma_estrogen_receptor_status",
  "patient.breast_carcinoma_progesterone_receptor_status",
  "patient.number_of_lymphnodes_positive_by_he",
  "patient.biospecimen_cqcf.tumor_samples.tumor_sample.tumor_weight",
  "patient.drugs.drug.therapy_ongoing"
)

brca_clinical <- survivalTCGA(BRCA.clinical, extract.cols = to_extract)
brca_mrna <- expressionsTCGA(BRCA.mRNA)

write.csv(brca_clinical, 'brca_clinical.csv')
write.csv(brca_mrna, 'brca_mrna.csv')
