
# 231109
# subset the genes by with top 3k variances

library(data.table)


# set the pwd of the files
countPwd = 'GSE67835_rawCount.csv'
labelPwd = 'GSE67835_cellLabel.csv'


# read the files into df
countDf = data.frame(fread(countPwd))
labelPwd = data.frame(fread(labelPwd))

rownames(countDf) = countDf$index
countDf$index = NULL

# calculate the variance of the genes
myVar = function(x){
  return(var(x[x>0]))
}

# varianceArr = apply(countDf, 1, var)
varianceArr = apply(countDf, 1, myVar)


# subset the genes by the top 3k genes

varianceCutoff = sort(varianceArr, decreasing = T)[3000]
countDfSubset = countDf[varianceArr >= varianceCutoff, ]



countDfSubset = countDf[rownames(countDf) %in% top10$gene, ]

dim(countDfSubset)

write.csv(countDfSubset, file = './GSE67835_rawCount_subset.csv',
          quote = F, row.names = T)



# draw the heatmap

countDfSubsetScaled = data.frame(scale(countDfSubset))
countDfSubsetScaled = na.omit(countDfSubsetScaled)

# draw the heatmap
library(pheatmap)
pheatmap(countDfSubsetScaled)



library(Seurat)
pbmc <- CreateSeuratObject(counts = countDf, project = "pbmc3k", min.cells = 3, min.features = 200)

pbmc


pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)


FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")



pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)


pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2



all.genes <- rownames(pbmc)
# pbmc <- ScaleData(pbmc, features = VariableFeatures(pbmc)) skylar's note: this did not keep
# relevant genes in scale.data
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))


# Examine and visualize PCA results a few different ways
print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)


VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")


DimHeatmap(pbmc, dims = 1, cells = 500, balanced = TRUE)



pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)



pbmc <- RunUMAP(pbmc, dims = 1:10)
# note that you can set `label = TRUE` or use the LabelClusters function to help label
# individual clusters
DimPlot(pbmc, reduction = "umap", label = T)


# find all markers of cluster 2
cluster2.markers <- FindMarkers(pbmc, ident.1 = 2)
head(cluster2.markers, n = 5)



# find markers for every cluster compared to all remaining cells, report only the positive
# ones
library(dplyr)
pbmc.markers <- FindAllMarkers(pbmc, only.pos = TRUE)

pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1) %>%
  # slice_head(n = 10) %>%
  slice_head(n = 10) %>%
  ungroup() -> top10

DoHeatmap(pbmc, features = top10$gene) + NoLegend()

