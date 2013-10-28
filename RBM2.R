sigmoid <- function(x) {
  1/(1+exp(-x))
}

RBM <- setRefClass(
  "RBM",
  
  fields = list (
    nV = "numeric",
    nH = "numeric",
    W = "matrix",
    vBias = "vector",
    hBias = "vector",
    seed = "numeric"
  ),
  
  methods = list (
    initialize = function(
      nH = 28*28,
      W = NULL,
      vBias = NULL,hBias = NULL,
      seed=100,
      nV = 28*28
    ) {
      nV <<- nV
      nH <<- nH
      seed <<- seed
      set.seed(seed)
      if (is.null(W)) {
        W <<- matrix(rep(0,nV*nH),nV,nH)
      } else {
        W <<- W
      }
      if (is.null(vBias)) {
        vBias <<- rep(0,nV)
      } else {
        vBias <<- vBias
      }
      if (is.null(hBias)) {
        hBias <<- rep(0,nH)
      } else {
        hBias <<- hBias
      }
    },
    
    #### conditional probavirities ####
    phv = function(visible) {
      preSig <- as.matrix(visible) %*% W + hBias
      list(p=sigmoid(preSig),preSig = preSig)
    },
    pvh = function(hidden) {
      'p(v|h)'
      preSig <- hidden %*% t(W) + vBias
      list(p=sigmoid(preSig),preSig = preSig)
    },
    
    #### gibbs sampling ####
    gibbsv2h = function(visible) {
      p <- phv(visible)$p
      #cat("vis input",dim(p),"\n")
      s<-rbinom(length(p),1,p)
      matrix(s,ncol=nH)
      #cat("vis output",dim(matrix(s,ncol=nH)),"\n")
    },
    gibbsh2v = function(hidden) {
      p <- pvh(hidden)$p
      #cat("hid input",dim(p),"\n")
      s<-rbinom(length(p),1,p)
      matrix(s,ncol=nV)
      #cat("hid output",dim(matrix(s,ncol=nV)),"\n")
    },
    gibbsv2h2v = function(visible) {
      gibbsh2v(gibbsv2h(visible))
    },
    gibbsh2v2h = function(hidden) {
      gibbsv2h(gibbsh2v(hidden))
    },
    
    #### updates params ####
    kcd = function(visible,k=1,lr = 0.1) {
      'k-contrastive divergence'
      v <- visible
      hidden <- gibbsv2h(visible)
      h <- hidden
      cat("kcd input size",dim(v),"\n")
      for (i in 1:k) {
        print("##### v2h2v #####")
        v <- gibbsv2h2v(v)
        print("##### h2v2h #####")
        h <- gibbsh2v2h(h)
      }
      dW <- (t(visible) %*% hidden - t(v) %*% h)/nrow(visible)
      dvBias <- apply((visible - v)/nrow(visible),2,sum)
      dhBias <- apply((hidden - h)/nrow(visible),2,sum)
      W <<- W + lr*dW
      vBias <<- vBias + lr*dvBias
      hBias <<- hBias + lr*dhBias  
    },
    
    #### train RBM ####
    # input:dataframe
    train = function(input,iter = 100, batchSize = 100, k=1, lr=0.1) {
      dataSize <- nrow(input)
      batchIter <- ceiling(dataSize/batchSize)
      for (i in 1:iter) {
        cat("iter",i,"\n")
        for (j in 1:(batchIter-1)) {
          cat("batch range",c((1+(j-1)*batchSize),j*batchSize),"\n")
          kcd(input[(1+(j-1)*batchSize):(j*batchSize),],k,lr)
        }
        cat("batch range",c((1+(batchIter-1)*batchSize),dataSize),"\n")
        kcd(input[(1+(batchIter-1)*batchSize):dataSize,],k,lr)
      }
    }
    
    #### draw filters ####
  )
)


INPUTENV <- new.env(TRUE,NULL)  # 新しい環境を作り tempenv と名付ける
get("INPUT",env=INPUTENV)  




