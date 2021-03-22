FROM r-base:3.6.1
RUN R -e "install.packages('data.table',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('dplyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('cluster',dependencies=TRUE, repos='http://cran.rstudio.com/')"
COPY ./Results /data
COPY clustering.R /
