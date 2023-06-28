# Generate a data frame that contains Cmax values for each GDSC drug where Cmax values are present

# Note: 
# Files "Liston_Davis_Drug_Concentrations_S1.txt" and "Liston_Davis_Drug_Concentrations_S2.txt" were obtained from https://doi.org/10.1158/1078-0432.c.6526332.v1
# File "screened_compunds_rel_8.2.csv" was downloaded from the GDSC website (https://www.cancerrxgene.org/downloads/bulk_download)

# get concentration data from paper (2 tables)
concs1 = read.csv("Liston_Davis_Drug_Concentrations_S1.txt", sep='\t', check.names = F)
concs2 = read.csv("Liston_Davis_Drug_Concentrations_S2.txt", sep='\t', check.names = F)
colnames(concs2)[colnames(concs2) == 'Cmax        (uM)'] = 'Cmax (uM)'
common_cols = c('Generic Name', 'Brand Name', 'Dose', 'Dose Unit', 'Cmax (uM)', 'Cmax (mol/liter)')
all_concs = rbind(concs1[, common_cols], concs2[, common_cols])

# get screened drugs from GDSC
screened_GDSC = read.csv("screened_compunds_rel_8.2.csv")

info_frame = data.frame('drug_id_GDSC'=screened_GDSC$DRUG_ID, 'drug_name_GDSC'=screened_GDSC$DRUG_NAME, 'synonyms_GDSC'=screened_GDSC$SYNONYMS, 'Generic Name'=NA, 'Brand Name'=NA, 'Dose'=NA, 'Dose Unit'=NA, 'Cmax (uM)'=NA, 'Cmax (mol/liter)'=NA, check.names=F)

for(i in 1:nrow(info_frame)) {
  drug = info_frame$drug_name_GDSC[i]
  data = all_concs[grepl(drug, all_concs$`Generic Name`) | grepl(drug, all_concs$`Brand Name`),]
  
  # handle special cases
  if(drug == 'Paclitaxel') { data = data[data$`Generic Name` == "Paclitaxel  ",] } # two types of Paclitaxel (Paclitaxel-Albumin und Paclitaxel)
  if(drug == 'Doxorubicin') { data = data[data$`Generic Name` == "Doxorubicin   ",] } # two types of Doxorubicin (liposomal and normal)
  if(drug == 'Irinotecan') { data = data[data$`Generic Name` == "Irinotecan   ",] } # two types of Irinotecan (liposomal and normal)
  if(drug == 'Vincristine') { data = data[data$`Generic Name` == 'Vincristine',] } # two types of Vincristine (liposomal and normal)
  if(drug == 'SN-38') {  # two types of SN-38 (liposomal and normal) and dose is given in different row
    data = data[data$`Generic Name` == '* SN-38',]
    info_frame$`Generic Name`[i] = data$`Generic Name`[1]
    info_frame$`Brand Name`[i] = all_concs$`Brand Name`[all_concs$`Generic Name` == "Irinotecan   "]
    info_frame$Dose[i] = all_concs$Dose[all_concs$`Generic Name` == "Irinotecan   "]
    info_frame$`Dose Unit`[i] = all_concs$`Dose Unit`[all_concs$`Generic Name` == "Irinotecan   "]
    info_frame$`Cmax (uM)`[i] = data$`Cmax (uM)`[1]
    info_frame$`Cmax (mol/liter)`[i] = data$`Cmax (mol/liter)`[1]
    next
  } 
  
  if(nrow(data) > 1) { cat(drug, " multiple hits! \n") }
  if(nrow(data) == 1) { info_frame[i, c(4:ncol(info_frame))] = as.list(data[1,])}
  
  # if drug was not found, search using synonyms
  if(nrow(data) == 0) {
    synonyms = info_frame$synonyms_GDSC[i]
    if(!is.na(synonyms)) {
      synonyms = strsplit(synonyms, ', ')[[1]]
      for(s in synonyms) {
        data = all_concs[grepl(s, all_concs$`Generic Name`) | grepl(s, all_concs$`Brand Name`),]
        
        if(drug == '5-Fluorouracil') { data = data[data$`Generic Name` == 'Fluorouracil  (5-FU)',] } # two types of 5-FU (one seems to be different drug based on Brand Name)
        
        if(nrow(data) > 1) { cat(drug, " multiple hits for synonym!\n") }
        if(nrow(data) == 1) { 
          info_frame[i, c(4:ncol(info_frame))] = as.list(data[1,])
          break
        }
      }
    }
  }
}

info_frame_filered = info_frame[!is.na(info_frame$`Cmax (uM)`),]
info_frame_filered$Cmax_ln = log(as.numeric(info_frame_filered$`Cmax (uM)`)) # one drug has string instead of Cmax -> Warning
info_frame_filered = na.omit(info_frame_filered) # remove the drug with no Cmax

write.table(info_frame_filered, "Liston_Davis_Cmax_for_GDSC_drugs.txt", sep='\t', row.names = F, quote = F)
