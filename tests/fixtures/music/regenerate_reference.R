# Regenerate one fixture using unmodified official MuSiC source functions.
# Rscript regenerate_reference.R /path/to/MuSiC input.json output.json
suppressPackageStartupMessages({library(nnls);library(SingleCellExperiment);library(jsonlite)})
a <- commandArgs(TRUE); root <- normalizePath(a[1]); a <- a[-1]
stopifnot(unname(tools::md5sum(file.path(root,'R/analysis.R'))) == 'e141a99984c12b67e7ccfc0a22877fdd')
stopifnot(unname(tools::md5sum(file.path(root,'R/construct.R'))) == '53a5037154fc69aadbddec9cbca1fa48')
stopifnot(unname(tools::md5sum(file.path(root,'R/utils.R'))) == '70b664006d1c5d206b3f8610a42efad0')
source(file.path(root,'R/analysis.R'));source(file.path(root,'R/construct.R'));source(file.path(root,'R/utils.R'))
z <- fromJSON(a[1]); X <- z$counts; B <- z$bulk
rownames(X)<-z$cells; colnames(X)<-z$genes;rownames(B)<-z$bulk_genes;colnames(B)<-z$bulk_samples
sce <- SingleCellExperiment(assays=list(counts=t(X)),colData=DataFrame(ct=z$types, donor=z$donors))
args <- list(bulk.mtx=B,sc.sce=sce,clusters='ct',samples='donor',verbose=FALSE,iter.max=z$max_iter)
if(!is.null(z$markers))args$markers<-z$markers
if(!is.null(z$cell_sizes))args$cell_size<-data.frame(ct=z$size_types,size=z$cell_sizes)
o<-do.call(music_prop,args)
write_json(o,a[2],digits=NA,na='null',auto_unbox=FALSE)
if(!is.null(z$group_cell_types)){
 colData(sce)$group<-z$groups
 rr<-music_prop.cluster(B,sce,group.markers=z$group_markers,groups='group',clusters='ct',samples='donor',clusters.type=z$group_cell_types,verbose=FALSE)
 write_json(rr,a[2],digits=NA,na='null',auto_unbox=FALSE)
}
