get_curve_function = function (maxc, xmid, scal) {
  fitted_curve = 
    function(lx) { 
      x = getXfromConc(exp(lx), maxc)
      return(1 - logist3(x, xmid, scal))
    }
  
  return(fitted_curve)
}

rsq = function(x, y) summary(lm(y~x))$r.squared

compute_cmax_viability = function(curves, database, table_path) {
  # only consider drugs with given Cmax
  curves = curves[curves$DRUG_ID_lib %in% unique(Cmax_data$drug_id_GDSC),]
  all_drugs = data.frame(drug_id_and_conc = unique(curves$drug), drug_name = NA, plot = F)
  
  for(d in 1:nrow(all_drugs)) {
    id = all_drugs$drug_id_and_conc[d]
    name = drug_info$DRUG_NAME[drug_info$DRUG_ID == gsub("_.*", "", id)]
    all_drugs$drug_name[d] = name
  }
  
  for(drug in 1:nrow(all_drugs)) {
    
    drug_id = all_drugs$drug_id_and_conc[drug]
    drug_name = all_drugs$drug_name[drug]
    drug_data = curves[curves$drug == drug_id,]
    
    if(nrow(drug_data) == 0) { 
      cat(drug_name, " skipped \n")
      next 
    }
    
    cat(drug, " ", drug_name, "\n")
    
    drug_CLs = unique(drug_data$CL)
    cmax = Cmax_data$Cmax_ln[Cmax_data$drug_id_GDSC == unique(drug_data$DRUG_ID_lib)]
    
    cmax_viability_frame = data.frame(cell_line = drug_CLs, cmax_viability = NA, IC50 = NA, RMSE = NA, R2 = NA, xmid = NA, scal = NA, maxc = NA, num_points = NA)
    
    for(i in 1:nrow(cmax_viability_frame)) {

      cl = cmax_viability_frame$cell_line[i]
      cl_data = drug_data[drug_data$CL == cl,]
      
      maxc = unique(round(cl_data$maxc,8))
      xmid = unique(round(cl_data$xmid,8))
      scal = unique(round(cl_data$scal,8))
      
      curve_fun = get_curve_function(maxc, xmid, scal)
      
      cl_data$lx = log(getConcFromX(cl_data$x, maxc))
      cl_data$viability = curve_fun(cl_data$lx)
      
      cmax_viability = curve_fun(cmax)
      
      cmax_viability_frame$cmax_viability[i] = cmax_viability
      cmax_viability_frame$IC50[i] = unique(round(cl_data$IC50, 8))
      cmax_viability_frame$RMSE[i] = unique(round(cl_data$RMSE, 8))
      cmax_viability_frame$R2[i] = rsq(cl_data$y, cl_data$yhat)
      cmax_viability_frame$xmid[i] = xmid
      cmax_viability_frame$scal[i] = scal
      cmax_viability_frame$maxc[i] = maxc
      cmax_viability_frame$num_points[i] = nrow(cl_data)
    }
    
    write.table(cmax_viability_frame, paste0(table_path, drug_name, "___", drug_id, "__Cmax_IC50_RMSE.txt"), sep = '\t', row.names = F, quote = F)
  }
}

### Read files

library(data.table)

# Note: 
# Files "GDSC1_fitted_curves_GDSCIC50.txt" and "GDSC2_fitted_curves_GDSCIC50.txt" were generated using Rscript "Compute_Dose_Response_Curves_gdscIC50.R"
# File "screened_compunds_rel_8.2.csv" was downloaded from the GDSC website (https://www.cancerrxgene.org/downloads/bulk_download)
# File "Liston_Davis_Cmax_for_GDSC_drugs.txt" was generated using Rscript "Get_Cmax_for_GDSC_drugs_from_Liston_Davis_Paper.R"

# Curves
GDSC1_curves = data.frame(fread("GDSC1_fitted_curves_GDSCIC50.txt", stringsAsFactors = F, sep = '\t'))
GDSC2_curves = data.frame(fread("GDSC2_fitted_curves_GDSCIC50.txt", stringsAsFactors = F, sep = '\t'))

# Drug names
drug_info = read.csv("screened_compunds_rel_8.2.csv", stringsAsFactors = F)

# Cmax data
Cmax_data = read.csv("Liston_Davis_Cmax_for_GDSC_drugs.txt", sep='\t')
Cmax_data$Drug_name_with_ID = paste0(Cmax_data$drug_name_GDSC, "___", Cmax_data$drug_id_GDSC)
Cmax_data$Cmax_ln = as.numeric(Cmax_data$Cmax_ln)

### Compute Cmax viabilities

library(gdscIC50)
library(ggplot2) 
library(ggpubr)
library(viridis)

# GDSC1
curves = GDSC1_curves
database = 'GDSC1'
table_path = "GDSC1_Tables/"
compute_cmax_viability_and_generate_viability_plots(curves, database, plot_drugs, table_path, plot_path)


# GDSC2
curves = GDSC2_curves
database = 'GDSC2'
table_path = "GDSC2_Tables/"
compute_cmax_viability(curves, database, table_path)


