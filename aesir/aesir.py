import scipy
import scipy.special as Sp
from numpy import *
import xmod
import time

class freyr:
    def __init__(self,data,K=100):
        self.data=data
        self.V=self.data[2].max()+1
        """We augment the feature indices in data[3] by one, reserving 0 for the absence of a feature"""
        self.F=self.data[3].max()+1-1
        self.J=self.data[0].max()+1
        self.nj=xmod.indsum(self.J,ascontiguousarray(self.data[0]),ascontiguousarray(ones(len(self.data[0])))).T[0]
        self.Nj=int(self.nj.sum())
        self.K=K

        self.theta=ones(self.K)/self.K
        self.beta=ones(self.V)/self.V
        self.gamma=ones(self.F)/self.F


        self.phi=clip(dirichletrnd(self.beta,self.K),1e-10,1-1e-10);
        self.psi=clip(dirichletrnd(self.gamma,self.K),1e-10,1-1e-10);
        self.pi=clip(dirichletrnd(self.theta,self.J),1e-10,1-1e-10);

        self.phiprior=dirichlet()
        self.psiprior=dirichlet()
        self.piprior=dirichlet()

        self.init_iteration_max=1e+2
        self.mcmc_iteration_max=1e+3
        self.iteration_eps=1e-5
        self.verbose=1
    
    def mcmc(self,level="one"):
        if level=="one":
            self.mcmc_level_one()
        #self.mcmc_functions[level]()

    def mcmc_level_one(self):
        iteration=0
        self.ll=empty(self.mcmc_iteration_max,float)

        while iteration<self.mcmc_iteration_max:
            self.fast_posterior()
            self.gamma_a_mle()
            self.theta_a_mle()
            self.beta_a_mle()

            self.ll[iteration]=self.pseudologlikelihood

            if self.verbose:
                print self.ll[iteration]
            iteration+=1
    


    def fast_posterior(self):

        vpsi=hstack(( ones((self.K,1)),self.psi))
        self.Rphi,self.Rpsi,self.S,Z=xmod.xfactorialposterior(self.phi,vpsi,self.pi,self.data,self.Nj,self.V,self.F+1,self.J,self.K)
        
        phi=clip(dirichletrnd_array(self.Rphi+self.beta),1e-10,1-1e-10);
        psi=clip(dirichletrnd_array(self.Rpsi[:,1:]+self.gamma),1e-10,1-1e-10);
        vpi=clip(dirichletrnd_array(self.S+self.theta),1e-10,1-1e-10)
        
        self.phi=ascontiguousarray((phi.T/phi.sum(1)).T)
        self.psi=ascontiguousarray((psi.T/psi.sum(1)).T)
        self.pi=ascontiguousarray((vpi.T/vpi.sum(1)).T)

        self.pseudologlikelihood=Z
        

    def slow_xposterior_fortestingonly(self):
        self.xposterior=zeros((self.K,self.Nj))
        for t in arange(self.Nj):
                f=zeros(self.K)
                for k in arange(self.K):
                    f[k]=log(self.pi[self.data[0,t],k]) + log(self.phi[k,self.data[2,t]])
                    if self.data[3,t]>0:
                        f[k]+= log(self.psi[k,self.data[3,t]-1])
                    
                for k in arange(self.K):
                    self.xposterior[k,t]=exp(f[k]-logsumexp(f))

        """    
        x_sample=multinomialrnd_array(x)
        
        Rphi=xmod.bigram(self.K,self.V,x_sample,ascontiguousarray(self.data[2]))
        Rpsi=xmod.bigram(self.K,self.F,x_sample,ascontiguousarray(self.data[3]))
        S=xmod.bigram(self.J,self.K,ascontiguousarray(self.data[0]),x_sample);
        """
        #self.phi=clip(dirichletrnd_array(Rphi+self.beta),1e-10,1-1e-10)
        #self.psi=clip(dirichletrnd_array(Rpsi+self.gamma),1e-10,1-1e-10)
        #self.pi=clip(dirichletrnd_array(S+self.theta),1e-10,1-1e-10)


    def beta_a_mle(self):
        self.phiprior.observation(self.phi)
        self.phiprior.a=self.beta.sum()
        self.phiprior.m=ones(self.V)/self.V
        self.phiprior.a_update()
        self.beta=self.phiprior.a*self.phiprior.m

    def theta_a_mle(self):
        self.piprior.observation(self.pi)
        self.piprior.a=self.theta.sum()
        self.piprior.m=ones(self.K)/self.K
        self.piprior.a_update()
        self.theta=self.piprior.a*self.piprior.m

    def gamma_a_mle(self):
        self.psiprior.observation(self.psi)
        self.psiprior.a=self.gamma.sum()
        self.psiprior.m=ones(self.F)/self.F
        self.psiprior.a_update()
        self.gamma=self.psiprior.a*self.psiprior.m
#
#    def getlatentlabels(self,k):
#        self.latent_labels=[]
#        s=fliplr(argsort(self.phi,axis=1,kind='mergesort'))
#        for i in xrange(len(s)):
#            a=[]
#            for j in s[i,:k+1]:
#                a.append((self.phi[i,j],self.data_labels[j]))
#            self.latent_labels.append(a)
#

    def getlatentlabels(self,k=10):
        self.latent_labels=[]
        for j in arange(self.K):
            Lphi=[]
            Lpsi=[]
            for i in flipud(argsort(self.phi[j])[-k:]):
                Lphi.append((self.vocab_labels[i],self.phi[j,i])),
            for i in flipud(argsort(self.psi[j])[-k:]):
                Lpsi.append((self.feature_labels[i],self.psi[j,i])),
            self.latent_labels.append((Lphi,Lpsi))

    def printlatentlabels(self,k=10):
        self.getlatentlabels(k)
        k=0
        for l in self.latent_labels:
            k+=1
            print str(k)+': ',
            for i in l[0]:
                print '%s(%2.2f)' % (i[0],i[1]),
            print "\n",
            for j in l[1]:
                print '%s(%2.2f)' % (j[0],j[1]),
            print "\n\n",
    
    def getfeaturelabels(self,file):
        self.feature_labels=open(file).read().split()
    
    def getvocablabels(self,file):
        self.vocab_labels=open(file).read().split()
    

class freyja:
    def __init__(self,data,K=100):
        
        self.data=ascontiguousarray(data)
        self.D=self.data[2].max()+1
        self.J=self.data[0].max()+1
        self.nj=xmod.indsum(self.J,ascontiguousarray(self.data[0]),ascontiguousarray(ones(len(self.data[0])))).T[0]
        self.K=K

        self.phi=dirichletrnd(ones(self.D),self.K)
        self.pi=dirichletrnd(ones(self.K),self.J)

        self.phiprior=dirichlet()
        self.piprior=dirichlet()

        #self.theta=ones(self.K)/self.K
        self.beta=ones(self.D)/self.D

        #self.gamma=random.random_sample(size=self.pi.shape)
        self.gamma=ones((self.J,self.K))
        self.gamma_b=ones((self.J,self.K))
        
        self.theta=random.rand(self.K)
        self.theta=self.theta/self.theta.sum()



        #self.beta=ones(self.D)

        self.init_iteration_max=1e+2
        self.mcmc_iteration_max=1e+3
        self.iteration_eps=1e-5
        self.verbose=1

        self.Nj=int(self.nj.sum())
        
        #self.maxarray=10**8
        self.maxarray=2.5*10**7

        """We run into memory problems here. 
        The posterior distribution over the latent variables for each observation will be a model.K*model.mj.sum() matrix
        To make this, intermediate matrices of similar size will be necessary
        To avoid memory exhaustion, we chop up this process into a few blocks
        The following calculates how many blocks (k) we need. 
        It checks if (model.K*model.nj.sum())/k < model.maxarray (which as of feb 7 2008 is set to 5*10**7)
        """
        k=1
        while (self.K*self.nj.sum())/k > self.maxarray:
            k+=1
        """ The next line just gets a set of indices that divide the range of 0 to model.J into k blocks """
        #print k
        self.range_index=hstack(((arange(0,k+1)*(self.J/k))[:-1],self.J))
        

    def initialize(self,level="one"):
        if level=="zero":
            self.initialization_level_zero()
        elif level=="one":
            self.initialization_level_one()

        #self.initialization_functions[level]()
        
    def initialization_level_zero(self):
        if self.verbose:
            print "initialization level zero"
          
        f=xmod.bigram(self.J,self.D,ascontiguousarray(self.data[0]),ascontiguousarray(self.data[2]))
        f=vstack((f,ones((1,self.D))))
        self.phi=dirichletrnd(f.mean(axis=0),self.K)
        
        self.pi=dirichletrnd(self.theta,self.J)

        self.beta_a_mle()
        self.theta_a_mle()
        

    def initialization_level_one(self):
        self.initialize("zero")

        if self.verbose: 
            print "initialization level one"
    
        self.ll=empty(self.init_iteration_max,float)
        iteration=0

        previous_loglike=-inf
        current_loglike=nan_to_num(-inf)
    
        while iteration<self.init_iteration_max and (1-current_loglike/previous_loglike) > self.iteration_eps:
            self.em()
            self.theta_a_mle()
            self.beta_a_mle()
            
            previous_loglike=current_loglike
            current_loglike=self.pseudologlikelihood
            if self.verbose:
                print current_loglike
            self.ll[iteration]=current_loglike
            iteration+=1

        
        if self.verbose:
            print self.pseudologlikelihood
    

    def mcmc(self,level="one"):
        if level=="one":
            self.mcmc_level_one()
        #self.mcmc_functions[level]()

    def mcmc_level_one(self):
        iteration=0
        self.ll=empty(self.mcmc_iteration_max,float)

        while iteration<self.mcmc_iteration_max:
            self.fast_posterior()
            self.theta_a_mle()
            self.beta_a_mle()
                
            self.ll[iteration]=self.pseudologlikelihood

            if self.verbose:
                print self.ll[iteration]
            iteration+=1



    def loglikelihood(self):

        qpi=self.pi
        #qpi=dirichletrnd(self.theta,self.J)

        iteration=0

        previous_loglike=-inf
        current_loglike=nan_to_num(-inf)
    
        while iteration<self.init_iteration_max and (1-current_loglike/previous_loglike) > self.iteration_eps:
            f=log(self.phi[:,self.data[2]]).T + log(qpi[self.data[0]])
            z=logsumexp(f)
            
            loglike=z.sum()

            qx=exp((f.T-z).T) # posterior over latent variables given everything

            g=xmod.indsum(self.J,ascontiguousarray(self.data[0]),qx)

            qpi=(g.T/g.sum(1)).T
            qpi=clip(qpi,1e-50,1-1e-50)
            
            previous_loglike=current_loglike
            current_loglike=loglike
    
            if self.verbose:
                print loglike

            iteration+=1
        
        zj=xmod.indsum(self.J,self.data[0],z).flatten()
        return (zj-log(sqrt(self.nj))*float(self.K-1)/2).sum()

    def logmarginal(self):
        z=[]
        for t in xrange(len(self.Phi)):
            self.phi=self.Phi[t]
            self.theta=self.Theta[t]
            self.pi=self.Pi[t]
            z.append(self.loglikelihood())

        #return z
        return log(len(z))-logharmonic(-array(z))

    def em(self):
        f=zeros((self.D,self.K))
        g=zeros((self.J,self.K))
        self.pseudologlikelihood=0
        """ We go through taking the ith and (i+1)th index of range_index and use this to extract out the blocks of the model.data matrix """
        for i,j in zip(self.range_index[:-1],self.range_index[1:]):
            self.ind=((i <= self.data[0] ) & (self.data[0] < j))
            F=log(self.phi[:,self.data[2,self.ind]]).T + log(self.pi[self.data[0,self.ind]])
            Z=logsumexp(F)
            x=exp((F.T-Z).T)
            f+=xmod.indsum(self.D,ascontiguousarray(self.data[2,self.ind]),x)
            g+=xmod.indsum(self.J,ascontiguousarray(self.data[0,self.ind]),x)
            self.pseudologlikelihood+=Z.sum()

        self.phi=clip((f/sum(f,axis=0)).T,1e-10,1-1e-10)
        self.pi= clip((g.T/sum(g,1)).T,1e-10,1-1e-10)

    def vem_estep(self):
        self.gamma=self.theta + ones((self.J,self.K))*(self.D/float(self.K))

        self.varpsi=zeros((self.K,int(self.nj.sum())))

        for t in arange(25):
            psi_gamma=xmod.psi(self.gamma)
            F=(exp(psi_gamma.T[:,self.data[0]])*self.phi[:,self.data[2]])
            self.varpsi=clip((F/F.sum(0)),1e-10,1-1e-10)
            
            prev_gamma=self.gamma.copy()
            self.gamma=self.theta+xmod.indsum(self.J,self.data[0],self.varpsi.T)

            if vdiff(self.gamma,prev_gamma)<1e-05:
                break
            

        self.variational_lower_bound()
        print self.vlb,self.theta.mean(),self.theta.sum()


    def vem_estep_b(self):

        self.gamma=self.theta + ones((self.J,self.K))*(self.D/float(self.K))

        self.varpsi=zeros((self.K,int(self.nj.sum())))
        for i in unique(self.data[0]):
            n=ones(self.K)*(self.D/float(self.K))
            for t in arange(100):
                f=self.phi[:,self.data[2,self.data[0]==i]].T*exp(xmod.psi(self.theta+n))
                q=f.T/f.sum(1)
                prev_n=n
                n=q.sum(1)
        
                self.varpsi[:,self.data[0]==i]=q

                if vdiff(n,prev_n)<1e-05:
                    break

            self.gamma[i]=self.theta+n

        self.varpsi=clip(self.varpsi,1e-10,1-1e-10)
        self.variational_lower_bound()
        print self.vlb,self.theta.mean(),self.theta.sum()

    
    def vem_mstep(self):
        """ My implementation of the Newton Step in Variational Inference in LDA """
        F=xmod.indsum(self.D,self.data[2],self.varpsi.T)
        self.phi=clip((F/F.sum(0)).T,1e-10,1-1e-10)

        psi_gamma=xmod.psi(self.gamma)
        psi_gamma_sum=xmod.psi(self.gamma.sum(1))

        self.S=psi_gamma.T-psi_gamma_sum
    
        init_theta=mean(self.gamma,0).sum()
        theta=init_theta

        K=float(self.K)

        theta_prev=theta.copy()
        
        iter=0
        while iter<100:
            
            if theta<=0:
                init_theta=init_theta/10.0
                #print "warning, warning... alpha is nan; new init is %2.2f" % init_theta
                theta=init_theta
                iter=0

            theta_prev=theta.copy()
            theta-=(self.J*(psi(theta)-psi(theta/K))+self.S.sum()/K)/(self.J*(Sp.polygamma(1,theta)-Sp.polygamma(1,theta/K)/K))
            
            if vdiff(theta,theta_prev)<1e-05:
                break
            
            
            #print t,theta

            #print self.J*(Sp.gammaln(theta)-self.K*Sp.gammaln(theta/self.K))+self.S.sum()*(theta/self.K-1)

        self.theta=theta*ones(self.K)/K

        #theta-=(model.J*(aesir.psi(theta)-aesir.psi((theta)/model.K))+model.S.sum()/model.K)/(model.J*(Sp.polygamma(1,theta)-Sp.polygamma(1,theta/model.K)/model.K))
        #K=float(model.K)

        #theta-=(model.J*(aesir.psi(theta)-aesir.psi(theta/K))+model.S.sum()/K)/(model.J*(Sp.polygamma(1,theta)-Sp.polygamma(1,theta/K)/K))
    
    def vem_mstep_b(self):
        
            """ Implementation of Newton Step in Mochashi Matlab LDA"""
            M,K=self.gamma.shape

            ini_alpha=mean(self.gamma,0)
            #print "new init_alpha is %2.2f" % ini_alpha.sum()
            g = zeros((1,K));
            pg=sum(xmod.psi(self.gamma),0)-sum(xmod.psi(sum(self.gamma,1)))

            g=zeros(self.K)
            alpha=ini_alpha.copy()
            palpha=zeros((1,self.K))
            
            maxiter=100
            iter=0

            while iter<100:
                alpha0=sum(alpha)

                g = self.J * (psi(alpha0)-xmod.psi(alpha)) + pg
                h = - 1 / xmod.tripsi(alpha)
                hgz = dot(h,g) / (1 / psi(alpha0,1) + sum(h))

                palpha=alpha.copy()
                for i in arange(self.K):
                    alpha[i]=alpha[i]-h[i]*(g[i]-hgz)/M
                
                #print alpha
                
                if any(alpha<0):
                    #print "warning, warning... alpha is nan; new init is %2.2f" % ini_alpha.sum()
                    ini_alpha/=10.0
                    alpha=ini_alpha.copy()
                    iter=0

                if vdiff(alpha,palpha)<1e-05:
                    break


            self.theta=alpha.copy()

    def vem_mstep_d(self):
        F=xmod.indsum(self.D,self.data[2],self.varpsi.T)
        self.phi=clip((F/F.sum(0)).T,1e-10,1-1e-10)

#        psi_gamma=xmod.psi(self.gamma)
#        psi_gamma_sum=xmod.psi(self.gamma.sum(1))
#
#        self.S=psi_gamma.T-psi_gamma_sum
#
#        a=.0001
#        b=100
#        for t in arange(10):
#            print a,b
#            THETA=linspace(a,b,100)
#            a,b=THETA[(self.J*(Sp.gammaln(THETA)-self.K*Sp.gammaln(THETA/self.K))+self.S.sum()*(THETA/self.K-1)).argsort()[-2:]]
#
#        
#        theta=THETA[(self.J*(Sp.gammaln(THETA)-self.K*Sp.gammaln(THETA/self.K))+self.S.sum()*(THETA/self.K-1)).argmax()]
#
#        self.theta=theta*ones(self.K)/self.K
#

    def vem_mstep_c(self):
        """ Implementation of Newton Step taken from Blei's C LDA program - I can't see how or why this should work as it doesn't follow the math"""
        F=xmod.indsum(self.D,self.data[2],self.varpsi.T)
        self.phi=clip((F/F.sum(0)).T,1e-10,1-1e-10)

        psi_gamma=xmod.psi(self.gamma)
        psi_gamma_sum=xmod.psi(self.gamma.sum(1))

        SS=(psi_gamma.T-psi_gamma_sum).sum()

        init_a=100.0
        log_a=log(init_a)
        
        for t in arange(25):
            a=exp(log_a)

            if isnan(a):
                init_a = init_a * 10
                print "warning, warning... alpha is nan; new init is %2.2f" % init_a 
                a=init_a
                log_a=log(a)

            f=self.J*(Sp.gammaln(a)-self.K*Sp.gammaln(a))+(a-1)*SS;
            df=self.J*(psi(a)-self.K*psi(a))+SS
            d2f=self.J*(Sp.polygamma(1,a)-self.K*Sp.polygamma(1,a))

            log_a = log_a - df/(d2f * a + df);
        
            print exp(log_a)


    def variational_lower_bound(self):
        psi_gamma=xmod.psi(self.gamma)
        psi_gamma_sum=xmod.psi(self.gamma.sum(1))

        S=psi_gamma.T-psi_gamma_sum

        self.vlb=self.J*(Sp.gammaln(self.theta.sum()) - Sp.gammaln(self.theta).sum()) \
        + dot(self.theta-1,S).sum() \
        + (self.varpsi*(S[:,self.data[0]])).sum() \
        + (self.varpsi*log(self.phi[:,self.data[2]])).sum() \
        - Sp.gammaln(self.gamma.sum(1)).sum() + Sp.gammaln(self.gamma).sum()\
        - ((self.gamma-1)*S.T).sum()\
        - (self.varpsi*log(self.varpsi)).sum()


    def posterior(self):

        R=zeros((self.K,self.D))
        S=zeros((self.J,self.K))
        self.pseudologlikelihood=0
        """ We go through taking the ith and (i+1)th index of range_index and use this to extract out the blocks of the model.data matrix """
        for i,j in zip(self.range_index[:-1],self.range_index[1:]):
            self.ind=((i <= self.data[0] ) & (self.data[0] < j))
            F=log(self.phi[:,self.data[2,self.ind]]).T + log(self.pi[self.data[0,self.ind]])
            Z=logsumexp(F)
            x=exp((F.T-Z).T)
            x_sample=multinomialrnd_array(x)

            R+=xmod.bigram(self.K,self.D,x_sample,ascontiguousarray(self.data[2,self.ind]))
            S+=xmod.bigram(self.J,self.K,ascontiguousarray(self.data[0,self.ind]),x_sample);
    
            self.pseudologlikelihood+=Z.sum()
        
        self.phi=clip(dirichletrnd_array(R+self.beta),1e-10,1-1e-10)
        self.pi=clip(dirichletrnd_array(S+self.theta),1e-10,1-1e-10)



    def fast_posterior(self):

        R,S,Z=xmod.xptest(self.phi,self.pi,self.data,self.Nj,self.D,self.J,self.K)

        phi=clip(dirichletrnd_array(R+self.beta),1e-10,1-1e-10);
        vpi=clip(dirichletrnd_array(S+self.theta),1e-10,1-1e-10)
        
        self.phi=ascontiguousarray((phi.T/phi.sum(1)).T)
        self.pi=ascontiguousarray((vpi.T/vpi.sum(1)).T)

        self.pseudologlikelihood=Z


    def slow_xposterior_fortestingonly(self):
        self.x=zeros(int(self.nj.sum()),dtype=int)
        for i in arange(self.nj.sum()):
            f=zeros(self.K)
            for k in arange(self.K):
                f[k]=log(self.phi[k,self.data[2,i]])+log(self.pi[self.data[0,i],k])

            m=max(f)
            z=m+log(exp(f-m).sum())
            p=exp(f-z)
            r=random.rand()
            x=0
            s=p[x]
            while not (r<s):
                x+=1
                s+=p[x]
            self.x[i]=x
            self.f=f
            self.p=p




    def sample(self,N,dt=1):
        self.Phi=zeros((N,)+self.phi.shape)
        self.Pi=zeros((N,)+self.pi.shape)
        self.Theta=zeros((N,)+self.theta.shape)
        self.ll=zeros(N)
        
        iteration=0
        t=0
        while iteration<N:
            self.xposterior()
            self.xsample()
            self.phi_posterior()
            self.pi_posterior()
            self.theta_posterior()
            
            t+=1
            if t==dt:
                t=0
                self.Phi[iteration]=self.phi
                self.Pi[iteration]=self.pi
                self.Theta[iteration]=self.theta
                self.ll[iteration]=self.pseudologlikelihood()
                if self.verbose:
                    print self.ll[iteration]
                iteration+=1

    def beta_mle(self):
        self.phiprior.observation(self.phi)
        #self.phiprior.a=self.beta.sum()
        #self.phiprior.m=ones(self.D)/self.D
        #self.phiprior.a_update()
        self.phiprior.initialize()
        self.phiprior.mle()
        self.beta=self.phiprior.a*self.phiprior.m


    def beta_a_mle(self):
        self.phiprior.observation(self.phi)
        self.phiprior.a=self.beta.sum()
        self.phiprior.m=ones(self.D)/self.D
        self.phiprior.a_update()
        #self.phiprior.initialize()
        #self.phiprior.mle()
        self.beta=self.phiprior.a*self.phiprior.m

    def theta_a_mle(self):
        self.piprior.observation(self.pi)
        self.piprior.a=self.theta.sum()
        self.piprior.m=ones(self.K)/self.K
        self.piprior.a_update()
        #self.phiprior.initialize()
        #self.phiprior.mle()
        self.theta=self.piprior.a*self.piprior.m


    def theta_mle(self):
        self.piprior.observation(self.pi)
        self.piprior.initialize()
        self.piprior.mle()
        self.theta=self.piprior.a*self.piprior.m

    def theta_posterior(self):
        self.piprior.observation(self.pi)
        self.piprior.initialize()
        self.piprior.mle()
        self.piprior.mcmc()
        self.theta=self.piprior.a*self.piprior.m
        
        # if the number of successful jumps is too small, then the stepsize is probably too big, so make it smaller
        if self.piprior.switch.mean()<.25:
            self.piprior.mcmc_stepsize/=2


    def read_data_labels(self,file):
        self.data_labels=open(file).read().split()

    def read_group_labels(self,file):
        self.group_labels=open(file).read().split()


    def __del__(self):
        """I want to die"""
    def info(self):
        print "I am a hierarchical mixture model, with latent-dimensionality of %d" % (self.K)
        


    def getlatentlabels(self,k):
        self.latent_labels=[]
        s=fliplr(argsort(self.phi,axis=1,kind='mergesort'))
        for i in xrange(len(s)):
            a=[]
            for j in s[i,:k+1]:
                a.append((self.phi[i,j],self.data_labels[j]))
            self.latent_labels.append(a)


    def printlatentlabels(self,k=10):
        self.getlatentlabels(k)
        for i in arange(len(self.latent_labels)):
            print  i+1, ":",
            for term in self.latent_labels[i]:
                print  '%s (%.2f)' % (term[1],term[0]),
            print  "\n",


def norm(x):
    return sqrt(sum(x**2))


def vdiff(n,p):
    return norm(n - p) / norm(n)


def slice_array_by_cols(p,ind):
    return ascontiguousarray(p[:,ind])


def logsumexp(A,axis_n=1):
    """ logsumexp - summing along rows """
    if A.ndim==1:
        M=A.max()
        return M+log(exp((A.T-M).T).sum())
    elif axis_n==1:
        M=A.max(axis=1)
        return M+log(exp((A.T-M).T).sum(axis=1))
    else:
        M=A.max(axis=0)
        return M+log(exp(A-M).sum(axis=0))


def logharmonic(ll):
    return log(len(ll))-logsumexp(-ll)

# some random number generators 
def dirichletrnd(a,J):
    g=random.gamma(a,size=(J,shape(a)[0]))
    return (g.T/sum(g,1)).T

def multinomialrnd(p,n):
    return argmax(random.uniform(0,1,(n,1))  <tile(cumsum(p),(n,1)),axis=1)


def multinomialrnd_array(p,N=1):
    """ Each row of p is a probability distribution. Draw single sample from each."""
    return argmax(random.uniform(0,1,(shape(p)[0],1))  <cumsum(p,1),1)
    """ to be able to sample N times from each distribution (i.e. each row of p) use the following (we think) """
    #return argmax(random.uniform(0,1,(N,p.shape[0],1))<tile(cumsum(p,1),(N,1,1)),2)


def dirichletrnd_array(a):
    g=random.gamma(a)
    return (g.T/sum(g,1)).T

def betarnd_array(a,b):
        a_sample=random.gamma(a)
        b_sample=random.gamma(b)
        return a_sample/(a_sample+b_sample)


# read data

def freya_dataread(file):

        data_file=open(file).readlines()
        w=[]

        for i in arange(len(data_file)):
            for x in data_file[i].split():
                y=x.split(":")
                w.append((int(y[0]),int(y[1]),i))

        return array(w).T

def dataread(file):
        # so far recognizes two data-types, lda and combinatorial lda
        data_file=open(file).readlines()
        
        mx=0    
        for group in data_file:
            for item in group.split():
                if len(item.split(":")[0].split(",")) > mx:
                    mx=len(item.split(":")[0].split(","))


        data=[]
        group_j=0
        for group in data_file:
            item_i=0
            for item in group.split():
                for i in xrange(int(item.split(":")[1])):
                    if mx==1:
                        # regular old lda format 
                        data.append([group_j,item_i,int(item.split(":")[0])])
                    elif mx==2:
                        # combinatorial lda format
                        x=[int(i) for i in  item.split(":")[0].split(",")]
                        
                        if len(x)==1:
                            # possible blanks, if so assume missing value is second one equal to zero
                            # we reserve 0 as the marker for blank, and so augment each index for second variable by 1
                            data.append([group_j,item_i,x[0],0])
                        else:
                            data.append([group_j,item_i,x[0],x[1]+1])
                    item_i+=1
            group_j+=1
        return array(data,int).T

def group_similarities_avg(model,k=10):
    
    P=zeros((model.J,model.J))
    for t in arange(len(model.Pi)):
        model.pi=model.Pi[t]
        norm_pi=model.pi/model.pi.sum(0)
        P+=inner(model.pi,norm_pi)

    # P[i][j] will be model.pi[i] dot times norm_pi[j]
    # so, each word is a row, each col is its associate
    for i in arange(len(P)):
        print '%s%s ' % (model.group_labels[i],":"), 
        p=P[i]
        s=flipud(argsort(p))
        for w in s[:k-1]:
            print '%s (%.2f)%s' % (model.group_labels[w], p[w],","),
        print '%s (%.2f)' % (model.group_labels[s[k]], p[s[k]])


def group_similarities(model,k=10):
    
    norm_pi=model.pi/model.pi.sum(0)
    P=inner(model.pi,norm_pi)
    # P[i][j] will be model.pi[i] dot times norm_pi[j]
    # so, each word is a row, each col is its associate
    for i in arange(len(P)):
        print '%s%s ' % (model.group_labels[i],":"), 
        p=P[i]
        s=flipud(argsort(p))
        for w in s[:k-1]:
            print '%s (%.2f)%s' % (model.group_labels[w], p[w],","),
        print '%s (%.2f)' % (model.group_labels[s[k]], p[s[k]])

        
class dirichlet:
    def __init__(self,K=10):
        self.K=K
        self.iteration_eps=1e-5
        self.iteration_max=10
        self.mcmc_stepsize=1e-1
        self.mcmc_iteration_max=25

    def observation(self,data):
        self.data=clip(data,1e-10,1-1e-10)
        self.J=data.shape[0]
        self.K=data.shape[1]
        self.logdatamean=log(self.data).mean(axis=0)
        
    def initialize(self):
        self.a,self.m=moment_match(self.data)
    
    def loglikelihood_gradient(self):
        return self.J*(psi(self.a)-psi(self.a*self.m)  + self.logdatamean)
    
    def loglikelihood(self):
        return self.J*(Sp.gammaln(self.a)-Sp.gammaln(self.a*self.m).sum()+dot(self.a*self.m-1,self.logdatamean))

    def a_new(self):
        d1=self.J*(psi(self.a) - dot(self.m,psi(self.a*self.m)) + dot(self.m,self.logdatamean));
        d2=self.J*(psi(self.a,1) - dot(self.m**2,psi(self.a*self.m,1)));
        self.a= (1/self.a+d1/d2/self.a**2)**-1

    def m_new(self):
        digamma_am= self.logdatamean-dot(self.m,self.logdatamean-psi(self.a*self.m))
        am=inv_digamma(digamma_am)
        self.m=am/sum(am)

    
    def a_update(self):
        a_old=self.a
        self.a_new()
        
        iteration=0
        
        while (abs(a_old-self.a)>self.iteration_eps) and iteration<self.iteration_max:
            a_old=self.a
            self.a_new()
            iteration+=1

    def m_update(self):
        m_old=self.m
        self.m_new()
        
        iteration=0

        while (abs(m_old-self.m).max()>self.iteration_eps) and iteration<self.iteration_max:
            m_old=self.m
            self.m_new()
            iteration+=1


    def mle(self):
        am_old=self.a*self.m
        self.a_update()
        self.m_update()

        iteration=0
        #print self.loglikelihood()

        while (abs(am_old-self.m*self.a).max()>self.iteration_eps) and iteration<self.iteration_max:
            am_old=self.a*self.m
            self.a_update()
            self.m_update()
            iteration+=1
            #print self.loglikelihood()
    
    def mcmc(self,mcmc_iteration_max=100):

        theta_current=self.a*self.m
        ll_current=self.loglikelihood()
        self.ll=zeros(mcmc_iteration_max)
        self.switch=zeros(mcmc_iteration_max)
        self.Theta=zeros((mcmc_iteration_max,self.K))

        iteration=0

        while iteration<mcmc_iteration_max:
            theta_proposed=theta_current*(1+random.uniform(-self.mcmc_stepsize,self.mcmc_stepsize,self.K))

            self.a=theta_proposed.sum()
            self.m=theta_proposed/self.a
            ll_proposed=self.loglikelihood()

            if exp(ll_proposed-ll_current)>random.rand():
                ll_current=ll_proposed
                self.switch[iteration]=1
                theta_current=theta_proposed

            self.ll[iteration]=ll_current
            self.Theta[iteration]=theta_current

            iteration+=1



def moment_match(data):
    """ Approximate the mean (m)  and precision (a)  of dirichlet distribution 
    by moment matching.
    m is mean(data,0)
    a is given by Ronning (1989) formula
    """
    m=data.mean(axis=0)
    s=log(m*(1-m)/var(data,0)-1).sum()
    return exp(s/(data.shape[1]-1)),m

def psi(x,d=0):
    if type(x)==ndarray:
        s=x.shape
        x=x.flatten()
        n=len(x)    
        
        y=empty(n,float)
        for i in xrange(n):
            y[i]=Sp.polygamma(d,x[i])
        
        return y.reshape(s)
    #elif type(x)==int or type(x)==float:
    else:
        return Sp.polygamma(d,x)



def inv_digamma(y,niter=5):
    x = exp(y)+1/2.0;
    Ind=(y<=-2.22).nonzero()
    x[Ind] = -1/(y[Ind] - psi(1));

    for iter in xrange(niter):
          x = x - (psi(x)-y)/psi(x,1);
    
    return x


class fricka:
    def __init__(self,D,K=50):
        self.D=D
        self.K=K

        self.phi=dirichletrnd(ones(self.D),self.K).T
        self.pi=dirichletrnd(ones(self.K),1).flatten()

        self.phiprior=dirichlet()
        self.piprior=dirichlet()

        self.theta=ones(self.K)
        self.beta=ones(self.D)

        self.verbose=1
        self.init_iteration_max=1e+3
        self.mcmc_iteration_max=1e+3
        self.iteration_eps=1e-5

    def observation(self,data):
        self.data=data
        self.J=self.data[0].max()+1
        self.nj=self.data.shape[1]

        x=[]
        y=[]
        for i in arange(self.nj):
            x.extend(self.data[1,i]*[self.data[0,i]])
            y.extend(self.data[1,i]*[self.data[2,i]])
        
        self.dataxt=vstack((array(x),array(y)))
        
#        self.nj=zeros(1+self.data[0].max())
#
#        for i in arange(len(self.data[0])):
#            if (self.data[1,i]+1)>self.nj[self.data[0,i]]:
#                self.nj[self.data[0,i]]=self.data[1,i]+1

    def test_observation(self,testdata):
        self.testdata=testdata

    def crossvalidate(self):
        f=log(self.phi[self.testdata[2],:]) + log(self.phi[self.testdata[0],:]) + log(self.pi)
        f=log(self.phi[self.data[0],:]) + log(self.phi[self.data[2],:]) + log(self.pi)
        return logsumexp(f).sum()

#    def xposterior(self):
#    
#        f=log(self.phi[self.data[0],:]) + log(self.phi[self.data[2],:]) + log(self.pi)
#        self.Z=logsumexp(f,1)
#        self.x=exp(f.T-self.Z).T


    def initialize(self,level="one"):
        if level=="zero":
            self.initialization_level_zero()
        elif level=="one":
            self.initialization_level_one()

        #self.initialization_functions[level]()
    
        
    def initialization_level_zero(self):
        if self.verbose:
            print "initialization level zero"
          
        f=xmod.bigram(self.J,self.D,ascontiguousarray(self.data[0]),ascontiguousarray(self.data[2]))
        self.phi=dirichletrnd(f.sum(axis=0)+1,self.K).T
        self.pi=dirichletrnd(ones(self.K),1)

        self.beta_posterior()
        #self.theta_posterior()

    def initialization_level_one(self):
        self.initialize("zero")

        if self.verbose: 
            print "initialization level one"
    

        self.ll=empty(self.init_iteration_max,float)
        iteration=0

        previous_loglike=-inf
        current_loglike=nan_to_num(-inf)
    
        while iteration<self.init_iteration_max and (1-current_loglike/previous_loglike) > self.iteration_eps:
            self.xposterior()
            self.mstep()
            
            previous_loglike=current_loglike
            current_loglike=self.loglikelihood()
            if self.verbose:
                print current_loglike
            self.ll[iteration]=current_loglike
            iteration+=1

        self.beta_posterior()
        #self.theta_posterior()

        if self.verbose:
            print self.loglikelihood()
    

    def mcmc(self,level="one"):
        if level=="one":
            self.mcmc_level_one()
        #self.mcmc_functions[level]()

    def mcmc_level_one(self):
        iteration=0
        self.ll=empty(self.mcmc_iteration_max,float)

        while iteration<self.mcmc_iteration_max:
            self.xposterior()
            #self.xsample()
            self.phi_posterior()
            #self.pi_posterior()
            
#            self.theta_posterior()
                
            self.ll[iteration]=self.loglikelihood()

            if self.verbose:
                print self.ll[iteration]
            iteration+=1


    def loglikelihood(self):
        return dot(self.Z,self.data[1])

    def xposterior(self):
    
        f=log(self.phi[self.data[0],:]) + log(self.phi[self.data[2],:]) + log(self.pi)
        self.Z=logsumexp(f,1)
        self.x=exp(f.T-self.Z).T

    
    def xsample(self):
        self.x_sample=multinomialrnd_array(self.x)


    def beta_posterior(self):
        self.phiprior.observation(self.phi.T)
        self.phiprior.initialize()
        self.phiprior.mle()
        self.beta=self.phiprior.a*self.phiprior.m


    def theta_posterior(self):
        self.piprior.observation(self.pi)
        self.piprior.initialize()
        self.piprior.mle()
#        self.piprior.mcmc()
        self.theta=self.piprior.a*self.piprior.m
        
        # if the number of successful jumps is too small, then the stepsize is probably too big, so make it smaller
#        if self.piprior.switch.mean()<.25:
#            self.piprior.mcmc_stepsize/=2


    def mstep(self):
        # I'm not sure I know what I am doing here, next two lines
        x=(self.x.T*self.data[1]).T
        f0=xmod.indsum(self.D,ascontiguousarray(self.data[2]),self.x)
        f1=xmod.indsum(self.D,ascontiguousarray(self.data[0]),self.x)
        f=f0+f1

        self.phi=clip((f/sum(f,axis=0)).T,1e-10,1-1e-10).T

        #g=xmod.indsum(self.J,ascontiguousarray(self.data[0]),self.x)
        #self.pi= clip((g.T/sum(g,1)).T,1e-10,1-1e-10)
        g=self.x.sum(0)
        self.pi=clip(g/sum(g),1e-10,1-1e-10)


    def phi_posterior(self):

        z=[]
        # the following loop could be hard coded in c
        for i in arange(self.nj):
            z.extend(multinomialrnd(self.x[i],self.data[1,i]))
            #z[i]=random.multinomial(self.data[1,i],self.x[i])

        self.z=array(z)

        R0=xmod.bigram(self.K,self.D,self.z,ascontiguousarray(self.dataxt[0]))
        R1=xmod.bigram(self.K,self.D,self.z,ascontiguousarray(self.dataxt[1]))

        self.R=R0+R1
        self.phi=clip(dirichletrnd_array(self.R+self.beta),1e-10,1-1e-10).T

#    def pi_posterior(self):
        self.S=xmod.unigram(self.K,self.z)
        self.pi=clip(dirichletrnd(self.S+self.theta,1).flatten(),1e-10,1-1e-10)
    


    def logmarginal(self):
        z=[]
        for t in xrange(len(self.Phi)):
            self.phi=self.Phi[t]
            self.pi=self.Pi[t]
            self.xposterior()
            z.append(self.loglikelihood()-log(self.nj)*float(self.K))
        #return z
        #return log(len(z))-logharmonic(-array(z))
        return logharmonic(array(z))


    def sample(self,N,dt=1):
        self.Phi=zeros((N,)+self.phi.shape)
        self.Pi=zeros((N,)+self.pi.shape)
        self.Theta=zeros((N,)+self.theta.shape)
        self.ll=zeros(N)
        
        iteration=0
        t=0
        while iteration<N:
            self.xposterior()
            #self.xsample()
            self.phi_posterior()
            #self.pi_posterior()
            
            t+=1
            if t==dt:
                t=0
                self.Phi[iteration]=self.phi
                self.Pi[iteration]=self.pi
                self.ll[iteration]=self.loglikelihood()
                if self.verbose:
                    print self.ll[iteration]
                iteration+=1



    def read_data_labels(self,file):
        self.data_labels=open(file).read().split()

    def read_group_labels(self,file):
        self.group_labels=open(file).read().split()

    


    def getlatentlabels(self,k=10):
        self.latent_labels=[]
        s=fliplr(argsort(self.phi.T,axis=1,kind='mergesort'))
        for i in xrange(len(s)):
            a=[]
            for j in s[i,:k+1]:
                a.append((self.phi[j,i],self.data_labels[j]))
            self.latent_labels.append(a)


    def printlatentlabels(self,k=10):
        self.getlatentlabels(k)
        for i in arange(len(self.latent_labels)):
            print i+1, ":",
            #print >> file, i+1, ":",
            for feature in self.latent_labels[i]:
                print '%s (%.2f)' % (feature[1],feature[0]),
                #print >> file, '%s (%.2f)' % (feature[1],feature[0])
            print "\n"
            #print >> file, "\n"

    def data_similarities(self,k=10):
        
        W=zeros((self.D,self.D))
        
        A=(self.phi*self.pi)
        P=A.T/A.sum(1)
        W+=dot(self.phi,P)
            
        # W: column_i is the prob dist over all w_j (for 1<j<D) given w_i 
        # W: W[j][i]=P(w_j|w_i)
        for i in arange(len(W)):
            print '%s%s ' % (self.data_labels[i],":"), 
            p=W[:,i]
            s=flipud(argsort(p))
            for w in s[:k-1]:
                print '%s (%.2f)%s' % (self.data_labels[w], p[w],","),
            print '%s (%.2f)' % (self.data_labels[s[k]], p[s[k]])


    def data_similarities_avg(model,k=10):
        
        W=zeros((self.D,self.D))
        for t in arange(len(self.Pi)):
            self.phi=self.Phi[t]
            self.pi=self.Pi[t]
            
            A=(self.phi*self.pi)
            P=A.T/A.sum(1)
            W+=dot(self.phi,P)
                
        for i in arange(len(W)):
            print '%s%s ' % (self.data_labels[i],":"), 
            p=W[:,i]
            s=flipud(argsort(p))
            for w in s[:k-1]:
                print '%s (%.2f)%s' % (self.data_labels[w], p[w],","),
            print '%s (%.2f)' % (self.data_labels[s[k]], p[s[k]])



def data_similarities(model,k=10):
    
    W=zeros((model.D,model.D))
    #for t in arange(len(model.Pi)):
    #t=0
    #model.phi=model.Phi[t]
    #model.pi=model.Pi[t]
    
    A=(model.phi.T*model.pi)
    P=A.T/A.sum(1)
    W+=dot(model.phi.T,P)
        
    # W: column_i is the prob dist over all w_j (for 1<j<D) given w_i 
    # W: W[j][i]=P(w_j|w_i)
    for i in arange(len(W)):
        print '%s%s ' % (model.data_labels[i],":"), 
        p=W[:,i]
        s=flipud(argsort(p))
        for w in s[:k-1]:
            print '%s (%.2f)%s' % (model.data_labels[w], p[w],","),
        print '%s (%.2f)' % (model.data_labels[s[k]], p[s[k]])


def get_test_data(data,p=.05):
    n=size(data,1)
    I=random.permutation(n)
    test_id=I[:round(n*p)]
    train_id=I[round(n*p):]
    return data[:,train_id],data[:,test_id]

#
#for i in arange(len(W)):
#    print '%s%s ' % (model.data_labels[i],":"), 
#    p=W[:,i]
#    s=flipud(argsort(p))
#    for w in s[:k-1]:
#        print '%s (%.2f)%s' % (model.data_labels[w], p[w],","),
#    print '%s (%.2f)' % (model.data_labels[s[k]], p[s[k]])


