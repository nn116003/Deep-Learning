source("RBM.R")
DBN <- setRefClass(
  "DBN",
  
  fields=list(
    l = "numeric"#層の数
    seed = "numeric",
    input = "data.frame",
    RBMs = "list",
    nOut = "numeric"#ラベルの数
    ),
  
  methods=list(
    initialize=function(
      nV,nHs=c(100,100), seed=0, nOut = 10
      ) {
      l <<- length(nHs)
      RBMs <<- list()
      node.nums <- c(nV,nHs)
      for (i in 1:l) {
        rbm <- RBM$new(nH = node.nums[i+1],nV = node.nums[i])
        RBMs <<- c(RBMs,rbm)
      }
    },
    
    #### train DBN ####
    # input:dataframe
    train = function(iter = 100, batchSize = 100, k=1, lr=0.1) {
      for (rbm in RBMs) {
        rbm$train(input,iter,batchSize,k,lr)
        input <- rbm$sample(input)
      }
    }
    )
  )