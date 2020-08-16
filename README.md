# 2L-SNM

elec.mat - The adjacency matrix of Wiki-election data.

epinion.txt- The edge list of Epinions data.

plfit.m, plplot.m, and plpva.m- are used to fit the power-law exponent of the given degree distribution, for plotting the degree distribution, and estimation of the goodness of fit (p-value) respectively. 

SignedNetRecalpha1.m- is used to generate model networks corresponding to 2L-SNM when data is Bitcoin-Alpha. Learned parameters are hardcoded in the program. 

  SignedNetRecepinion1.m- is used to generate model networks corresponding to 2L-SNM when data is Epinions.  Learned parameters are hardcoded in the program. 

SignedNetRecOTC1.m- is used to generate model networks corresponding to 2L-SNM when data is Bitcoin-OTC.  Learned parameters are hardcoded in the program. 

SignedNetRecslashdot1.m- is used to generate model networks corresponding to 2L-SNM when data is Slashdot.  Learned parameters are hardcoded in the program. 

SignedNetRecwikielection1.m- is used to generate model networks corresponding to 2L-SNM when data is Wiki-election. Learned parameters are hardcoded in the program.   

slashdot.txt- The edge list of Slashdot data.

soc-sign-bitcoin.mat- The adjacency matrix of Bitcoin-OTC data.

  soc-sign-bitcoinalpha.mat- The adjacency matrix of Bitcoin-Alpha data.

two_SNM_A.m- is a model parameter trainer if the adjacency matrix of a real-world signed network is given.  It is used to learn the model parameters of  Bitcoin-Alpha, Bitcoin-OTC, and Wiki-election.

  two_SNM_Elist.m- is a model parameter trainer for Slashdot using the edge list.

  two_SNM_Elist1.m- is a model parameter trainer for Epinions using the edge list. 
