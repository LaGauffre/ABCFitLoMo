#Retrieving some insurance datasets from the package CASDatasets

#Getting the files from Christophe webpage
install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")
library(CASdatasets)
library(readr)

#Getting an australian motor insurance dataset
data(ausautoBI8999) 
ausautoBI8999_data <- data.frame(ausautoBI8999)
write_csv(ausautoBI8999_data,
          "C:/Users/pierr/Dropbox/Goffard-Laub/Code/Data/ausautoBI8999.csv",
          append = FALSE, col_names = TRUE)

#Getting a Belgian automobile claim dataset
data("besecura")
besecura_data <- data.frame(besecura)
write_csv(besecura_data,
          "C:/Users/pierr/Dropbox/Goffard-Laub/Code/Data/besecura.csv",
          append = FALSE, col_names = TRUE)
#Italian motor insurance data set
data(itamtplcost)
itamtpcost_data <- data.frame(itamtplcost)
write_csv(itamtpcost_data,
          "C:/Users/pierr/Dropbox/Goffard-Laub/Code/Data/itamtpcost_data.csv",
          append = FALSE, col_names = TRUE)
