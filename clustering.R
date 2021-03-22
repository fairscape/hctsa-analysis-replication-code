library(data.table)
vectorized_pdist <- function(A,B){
  an = apply(A, 1, function(rvec) crossprod(rvec,rvec))
  bn = apply(B, 1, function(rvec) crossprod(rvec,rvec))
  m = nrow(A)
  n = nrow(B)
  tmp = matrix(rep(an, n), nrow=m)
  tmp = tmp +  matrix(rep(bn, m), nrow=m, byrow=TRUE)
  return(as.vector(sqrt( tmp - 2 * tcrossprod(A,B) )))
}
library(dplyr)
setwd('/Path/to/Data/')
df = fread('./normalized_data.csv', header = TRUE)

#Have to calculate distance between all points many more points
#takes really long time
test <- df %>% group_by(id)

#Drop columns not included in clustering
drops <- c('DN_Observations','id','time')
test <- test[ , !(names(test) %in% drops)]
rm(df)

#Change all names to Vital.Group.Algo scheme
names(test) <- gsub(" ",".",names(test))
names(test) <- gsub("\\[",".",names(test))
names(test) <- gsub("\\'",".",names(test))
names(test) <- gsub("\\,",".",names(test))
names(test) <- gsub("]",".",names(test))

#Distance between all points
yikes <- vectorized_pdist(as.matrix(test),as.matrix(test))
yikes <- yikes[seq(1, length(yikes), 10)]
yikes[is.na(yikes)] = 0


dis <- read.csv('./distance_between_operations_HR_SP.csv')
library(cluster)

library(Matrix)
rowscols <- which(upper.tri(dis), arr.ind=TRUE)
dis <- sparseMatrix(i=rowscols[,1],    #rows to fill in
                    j=rowscols[,2],    #cols to fill in
                    x=dis[upper.tri(dis)], #values to use (i.e. the upper values of e)
                    symmetric=TRUE,    #make it symmetric
                    dims=c(nrow(dis),nrow(dis))) #dimensions
dis <- as.data.frame(as.matrix(dis))

names(dis) <- names(test)
names(dis) <- gsub(" ",".",names(dis))
names(dis) <- gsub("\\[",".",names(dis))
names(dis) <- gsub("\\'",".",names(dis))
names(dis) <- gsub("\\,",".",names(dis))
names(dis) <- gsub("]",".",names(dis))
rownames(dis) <- names(dis)

#for number of clusters of interest
#calculates lots of metrics about each set of clusters
#Residual variance describes how close to full dataset reduced features were.
clusters = c(1,2,3,4,5,10,15,20,25,30,35,40,45,50)
for ( i in clusters){#seq(from = 1, to = 10, by = 1)){
  print(i)
  result <- pam(dis,i,diss = TRUE)


  reduced_df <- test[,result$medoids]

  reduced_dist <- vectorized_pdist(as.matrix(reduced_df),as.matrix(reduced_df))

  reduced_dist <- reduced_dist[seq(1, length(reduced_dist), 10)]
  reduced_dist[is.na(reduced_dist)] = 0
  result$residualVariance = 1 - cor(yikes,reduced_dist)^2
  saveRDS(result, file = paste("Cluster Sampled ",as.character(i), "Results.rds",sep=" "))

}
