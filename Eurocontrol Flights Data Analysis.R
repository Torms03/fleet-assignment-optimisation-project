install.packages(readxl)
install.packages(tidyverse)
install.packages(modeldata)

library(readxl)
library(tidyverse) 
library(modeldata)
library(ggplot2)
library(dplyr)

flights_data <- read_excel('Flights_Data.xlsx')

str(flights_data)

boxplot(flights_data$`Actual Distance Flown (nm)`)

sort_data <- sort(flights_data$`Actual Distance Flown (nm)`)

sort_data

ggplot(flights_data, aes(x = `AC Operator`)) + 
  geom_bar()
        
#https://favtutor.com/blogs/frequency-table-in-r
freq_ac_operator <- flights_data %>%
  group_by(`AC Operator`) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency))

freq_ac_operator

#https://www.rpubs.com/dvdunne/reorder_ggplot_barchart_axis
# Barplot
ggplot(head(freq_ac_operator, n=20), aes(x= reorder(`AC Operator`,-frequency), y=frequency)) + 
  geom_bar(stat = "identity")

#https://www.caa.co.uk/commercial-industry/airports/aerodrome-licences/certificates/uk-certificated-aerodromes/
uk_aerodromes <- c('EGPD','EGNM','EGAA','EGGP','EGAC','EGLC','EGPL','EGKK','EGBB','EGLL','EGHH','EGGW','EGGD','EGSS','EGEC','EGCC','EGSC','EGNT','EGFF','EGDG','EGAE','EGSH','EGPN','EGTK','EGNX','EGPK','EGPH','EGHI','EGTE','EGMC','EGPF','EGPO','EGNR','EGPB','EGNJ','EGNV','EGPE','EGPU','EGPI','EGPC','EGPA')

uk_flights_data <- subset(flights_data,(flights_data$ADEP %in% uk_aerodromes | flights_data$ADES %in% uk_aerodromes))

uk_flights_data

uk_freq_ac_operator <- uk_flights_data %>%
  group_by(`AC Operator`) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency))

uk_freq_ac_operator

ggplot(head(uk_freq_ac_operator, n=10), aes(x= reorder(`AC Operator`,-frequency), y=frequency)) + 
  geom_bar(stat = "identity")

uk_flights_data$ADEP_ADES <- with(uk_flights_data, paste(uk_flights_data$ADEP,uk_flights_data$ADES, sep = '-'))

uk_flights_data

uk_BA_flights_data <- uk_flights_data[uk_flights_data$`AC Operator`== 'BAW',]

uk_BA_freq_dest <- uk_BA_flights_data %>%
  group_by(`ADEP_ADES`) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency))

uk_BA_freq_dest

ggplot(head(uk_BA_freq_dest, n=10), aes(x= reorder(`ADEP_ADES`,-frequency), y=frequency)) + 
  geom_bar(stat = "identity")


uk_freq_ac_type <- uk_flights_data %>%
  group_by(`AC Type`) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency))

uk_freq_ac_type

ggplot(head(uk_freq_ac_type, n=10), aes(x= reorder(`AC Type`,-frequency), y=frequency)) + 
  geom_bar(stat = "identity")

uk_BA_freq_ac_type <- uk_BA_flights_data %>%
  group_by(`AC Type`) %>%
  summarise(frequency = n()) %>%
  arrange(desc(frequency))

uk_BA_freq_ac_type

ggplot(head(uk_BA_freq_ac_type, n=10), aes(x= reorder(`AC Type`,-frequency), y=frequency)) + 
  geom_bar(stat = "identity")
