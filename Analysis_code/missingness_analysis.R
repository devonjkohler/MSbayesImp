
library(MSstats)
library(tidyverse)
library(data.table)
library(betareg)

## DDA Choi MaxQ
evidence = read.csv("DDA_Choi2017/Choi2017_DDA_MaxQuant_evidence.txt", sep="\t")
protgroups = read.csv("DDA_Choi2017/Choi2017_DDA_MaxQuant_proteinGroups.txt", sep="\t")
annotation = read.csv("DDA_Choi2017/Choi2017_DDA_MaxQuant_annotation.csv")

input = MaxQtoMSstatsFormat(evidence, annotation, protgroups)
input$feature = paste(input$ProteinName, input$PeptideSequence, 
                      input$PrecursorCharge, sep="_")

missing_feat = input %>% group_by(feature) %>% summarize(
    missing_percent = sum(is.na(Intensity)) / n(), 
    avg_intensity = log2(mean(Intensity, na.rm=TRUE)))
input = input %>% merge(missing_feat, all.x=TRUE, by="feature")

as.data.frame(input) %>% ggplot() + geom_point(
    aes(x=log2(Intensity), y=missing_percent), alpha=.25, position="jitter")

missing_feat %>% ggplot() + geom_point(
    aes(x=log2(avg_intensity), y=missing_percent), alpha=.25, position="jitter")

model = glm(missing_percent~avg_intensity, data=missing_feat, family="quasibinomial")
summary(model)

missing_feat %>% 
    ggplot() + geom_histogram(aes(missing_percent))


missing_feat %>% ggplot() + geom_point(
    aes(x=avg_intensity, y=missing_percent), alpha=.25, position="jitter") + 
    geom_line(data=data.frame(x=seq(15, 30, by = 0.5),
                              y=predict(model, data.frame(avg_intensity=seq(15, 30, by = 0.5)), type="response")), 
              aes(x=x, y=y), color="red")


missing_feat = input %>% group_by(ProteinName) %>% summarize(
    missing_percent = sum(is.na(Intensity)) / n(), 
    avg_intensity = log2(mean(Intensity, na.rm=TRUE)))

model = glm(missing_percent~avg_intensity, data=missing_feat, family="quasibinomial")
missing_feat %>% ggplot() + geom_point(
    aes(x=avg_intensity, y=missing_percent), alpha=.25, position="jitter") + 
    geom_line(data=data.frame(x=seq(15, 30, by = 0.5),
                              y=predict(model, 
                                        data.frame(avg_intensity=seq(15, 30, by = 0.5)), 
                                        type="response")), 
              aes(x=x, y=y), color="red")


## DDA Meier MaxQ
evidence = read.csv("DDA_Meierhofer2016/evidence.txt", sep="\t")
protgroups = read.csv("DDA_Meierhofer2016/proteinGroups.txt", sep="\t")
annotation = read.csv("DDA_Meierhofer2016/experimentalDesign_quant_annotation.csv")

meier_input = MaxQtoMSstatsFormat(evidence, annotation, protgroups)
meier_input$feature = paste(meier_input$ProteinName, meier_input$PeptideSequence, 
                            meier_input$PrecursorCharge, sep="_")

missing_feat = meier_input %>% group_by(feature) %>% summarize(
    missing_percent = sum(is.na(Intensity)) / n(), 
    avg_intensity = log2(mean(Intensity, na.rm=TRUE)))

model = glm(missing_percent~avg_intensity, data=missing_feat, family="quasibinomial")
plot(model, 2)

missing_feat %>% ggplot() + geom_point(
    aes(x=avg_intensity, y=missing_percent), alpha=.25, position="jitter") + 
    geom_line(data=data.frame(x=seq(18, 33, by = 0.5),
                              y=predict(model, data.frame(avg_intensity=seq(18, 33, by = 0.5)), type="response")), 
              aes(x=x, y=y), color="red")

missing_feat = meier_input %>% group_by(ProteinName) %>% summarize(
    missing_percent = sum(is.na(Intensity)) / n(), 
    avg_intensity = log2(mean(Intensity, na.rm=TRUE)))

model = glm(missing_percent~avg_intensity, data=missing_feat, family="quasibinomial")
plot(model, 2)
missing_feat %>% ggplot() + geom_point(
    aes(x=avg_intensity, y=missing_percent), alpha=.25, position="jitter") + 
    geom_line(data=data.frame(x=seq(15, 35, by = 0.5),
                              y=predict(model, data.frame(avg_intensity=seq(15, 35, by = 0.5)), type="response")), 
              aes(x=x, y=y), color="red")

