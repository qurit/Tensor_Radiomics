# Tensor Radiomics:  Discretization flavours 
By Natalia Dubljevic (@pasalacquaian) and Amirhossein Toosi (@Amirhosein2c)

The proposed TR approach primarily leverages explainable, handcrafted radiomics features at different discretization levels. Discretization involves grouping the original range of pixel intensities into specific intervals or bins, which is essential for computational feasibility of certain features. For instance, this can be achieved by fixing the bin widths (BW) or the number of bins (bin counts (BC)).

* conventional_baseline: This trains and tests on conventional pipelines using single-flavour features
* tr_flavour_concatenation: This trains and tests the flavour concatenation TR implementation using the same pipelines as the baseline
* tr_flavour_combination: This trains and tests the flavour combination TR implementation using mostly the same pipelines as the baseline. PCA as a dimensionality reduction technique is dropped for interpretability.
* tr_net: Trains and tests TR-Net

