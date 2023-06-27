library(gdscIC50)
library(data.table)

# NOTE: files "GDSC1_public_raw_data_25Feb20.csv" and "GDSC2_public_raw_data_25Feb20.csv" were obtained from GDSC website (https://www.cancerrxgene.org/downloads/bulk_download)

### Preprocessing ###

# GDSC1: negative controls are NC-0 (media only) (https://www.cancerrxgene.org/help#t_curve)
gdsc1 = data.frame(fread("GDSC1_public_raw_data_25Feb20.csv"))
gdsc1_no_fails = removeFailedDrugs(gdsc1)
gdsc1_no_missing_drugs = removeMissingDrugs(gdsc1_no_fails)
gdsc1_normalized = normalizeData(gdsc1_no_missing_drugs, trim = T, neg_control = "NC-0", pos_control = "B")
write.table(gdsc1_normalized, "GDSC1_relative_viabilities.txt", row.names = F, quote = F, sep='\t')

cat("GDSC1 preprocessing done! \n")

# GDSC2: negative controls are NC-1 (media + DMSO)
gdsc2 = data.frame(fread("GDSC2_public_raw_data_25Feb20.csv"))
gdsc2_no_fails = removeFailedDrugs(gdsc2)
gdsc2_no_missing_drugs = removeMissingDrugs(gdsc2_no_fails)
gdsc2_normalized = normalizeData(gdsc2_no_missing_drugs, trim = T, neg_control = "NC-1", pos_control = "B")
write.table(gdsc2_normalized, "GDSC2_relative_viabilities.txt", row.names = F, quote = F, sep='\t')

cat("GDSC2 preprocessing done! \n")


### Curve-Fitting ###

cat('GDSC1 reading ... \n')
gdsc1_viability = data.frame(fread("GDSC1_relative_viabilities.txt", stringsAsFactors = F, sep = '\t'))
cat('GDSC1 fitting ... \n')
after_conc_setting = setConcsForNlme(gdsc1_viability, group_conc_ranges = F)
after_prepping = prepNlmeData(after_conc_setting, cl_id = "COSMIC_ID")
model = fitModelNlmeData(after_prepping, isLargeData = T)
stats = calcNlmeStats(model, after_prepping)
cat('GDSC1 writing ... \n')
write.table(stats, "GDSC1_fitted_curves_GDSCIC50.txt", row.names = F, quote = F, sep = '\t')

cat("GDSC1 done! \n")

cat('GDSC2 reading ... \n')
gdsc2_viability = data.frame(fread("GDSC2_relative_viabilities.txt", stringsAsFactors = F))
cat('GDSC2 fitting ... \n')
after_conc_setting = setConcsForNlme(gdsc2_viability, group_conc_ranges = F)
after_prepping = prepNlmeData(after_conc_setting, cl_id = "COSMIC_ID")
model = fitModelNlmeData(after_prepping, isLargeData = T)
stats = calcNlmeStats(model, after_prepping)
write.table(stats, "GDSC2_fitted_curves_GDSCIC50.txt", row.names = F, quote = F, sep = '\t')
cat('GDSC2 writing ... \n')

cat("GDSC2 done! \n")


