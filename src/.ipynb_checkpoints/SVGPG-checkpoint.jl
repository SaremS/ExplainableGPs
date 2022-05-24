using Distributions, DistributionsAD, LinearAlgebra, KernelFunctions, BlockDiagonals, MLDatasets, Zygote, LinearAlgebra, Flux
import Base.+, Base.-, Base.*


struct Gaussian
    μ
    Σ
end
Flux.@functor Gaussian


+(x::Gaussian, y::Gaussian) = Gaussian(x.μ .+ y.μ, x.Σ .+ y.Σ)
+(x::Gaussian, y::Vector) = Gaussian(x.μ .+ y, x.Σ)
+(x::Vector, y::Gaussian) = Gaussian(x .+ y.μ, y.Σ)

-(x::Gaussian, y::Gaussian) = Gaussian(x.μ .- y.μ, x.Σ .+ y.Σ)
-(x::Gaussian, y::Vector) = Gaussian(x.μ .- y, x.Σ)
-(x::Vector, y::Gaussian) = Gaussian(x .- y.μ, y.Σ)

*(x::Matrix, y::Gaussian) = Gaussian(x*y.μ, (x)*y.Σ*transpose(x))
        
*(x::Gaussian, y::Real) = Gaussian(y.*x.μ, y^2 .*x.Σ)
*(x::Real, y::Gaussian) = Gaussian(x.*y.μ, x^2 .*y.Σ)

Gaussian(m::Gaussian, S) = Gaussian(m.μ, S.+m.Σ)
Distributions.MvNormal(m::Gaussian) = Distributions.MvNormal(m.μ, Matrix(Hermitian(m.Σ.+Diagonal(ones(length(m.μ)).*1e-6))))
            



struct SEKernel <: KernelFunctions.Kernel
     
    se_variance
    se_lengthscale

end

Flux.@functor SEKernel


SEKernel() = SEKernel(zeros(1,1), zeros(1,1))


function KernelFunctions.kernelmatrix(m::SEKernel,x::Matrix,y::Matrix)

    diffed = sum((Flux.unsqueeze(x,3) .- Flux.unsqueeze(y,2)).^2,dims=1)[1,:,:]
    
    return exp(m.se_variance[1,1]) .* exp.(- 0.5 *exp(m.se_lengthscale[1,1]) .* diffed)

end

KernelFunctions.kernelmatrix(m::SEKernel,x::Matrix) = KernelFunctions.kernelmatrix(m,x,x)


function crossderivativekernel(m::SEKernel,x::Matrix,y::Matrix,i=1)

    diffed = (Flux.unsqueeze(x[i:i,:],3) .- Flux.unsqueeze(y[i:i,:],2))[1,:,:]
    kern = kernelmatrix(m,x,y)
    
    result =  exp(m.se_lengthscale[1,1]) .* diffed .* kern
    
    return result
    
end


crossderivativekernel(m::SEKernel,x::Matrix) = crossderivativekernel(m,x,x)
crossderivativekernel(m::SEKernel,x::Matrix,i::Int) = crossderivativekernel(m,x,x,i)


function doaug(k,n,m)
    
    result = zeros(k,k,n,m,k*n,k*m)
    
    for i in 1:k
        for j in 1:k
            for s in 1:n
                for t in 1:m
                    result[i,j,s,t,((i-1)*n+s),((j-1)*m+t)] = 1.
                end
            end
        end
    end
    
    return result
end
Zygote.@nograd doaug


function derivativekernel(m::SEKernel, x::Matrix, y::Matrix,i=1)
    
    diffed = (Flux.unsqueeze(x[i:i,:],3) .- Flux.unsqueeze(y[i:i,:],2))[1,:,:]

    kern = kernelmatrix(m,x,y)
    
    result =  (exp(m.se_lengthscale[1,1]) .- exp(m.se_lengthscale[1,1]).^2 .* diffed.^2) .* kern
        
    return result
end


derivativekernel(m::SEKernel,x::Matrix) = derivativekernel(m,x,x)
derivativekernel(m::SEKernel,x::Matrix,i::Int) = derivativekernel(m,x,x,i)



struct SVGPG
     
    L
    a
    I
    
    kern
    s
    
end
Flux.@functor SVGPG



SVGPG(X, kern, ndims=1, nind=5) = SVGPG(randn(nind,nind), randn(1,nind), X[:,1:nind], kern, zeros(1,1))

SVGPG(kern, ndims=1, nind=5) = SVGPG(randn(nind,nind), randn(1,nind), rand(ndims,nind).*6 .-3, kern, zeros(1,1))


function (m::SVGPG)(x)
    
    S = m.L*transpose(m.L)
    
    Kmm = kernelmatrix(m.kern,m.I)
    Kmn = kernelmatrix(m.kern,m.I,x)
    Knn = kernelmatrix(m.kern,x)
    
    L = cholesky(Kmm.+Diagonal(ones(size(Kmm,2)).*exp(m.s[1,1]))).L
    v = L\Kmn


    means = transpose(transpose(L)\(L\transpose(m.a)))*Kmn
    covs =  Knn .- transpose(v)*v
    
    return Gaussian(means[:], covs)

end



struct ∂SVGPG
    
    SVGP
    
end
Flux.@functor ∂SVGPG


∂(m::SVGPG) = ∂SVGPG(m)

function (mm::∂SVGPG)(x,i::Int=1)
    
    m = mm.SVGP
    
    S = m.L*transpose(m.L)
    
    Kmm = kernelmatrix(m.kern,m.I)
    Kmn = crossderivativekernel(m.kern,m.I,x,i)
    Knn = derivativekernel(m.kern,x,i)
    
    L = cholesky(Kmm.+Diagonal(ones(size(Kmm,2)).*exp(m.s[1,1]))).L
    v = L\Kmn


    means = transpose(transpose(L)\(L\transpose(m.a)))*Kmn
    covs =  Knn .- transpose(v)*v
    
    return Gaussian(means[:], covs)

end





struct CrossSVGPG
    
    SVGP
    
end
Flux.@functor ∂SVGPG


cross(m::SVGPG) = CrossSVGPG(m)

function (mm::CrossSVGPG)(x,i::Int=1)
    
    m = mm.SVGP
    
    S = m.L*transpose(m.L)
    
    Kmm = kernelmatrix(m.kern,m.I)
    Kmn = crossderivativekernel(m.kern,m.I,x,i)
    Knn = crossderivativekernel(m.kern,x,i)
    
    L = cholesky(Kmm.+Diagonal(ones(size(Kmm,2)).*exp(m.s[1,1]))).L
    v = L\Kmn


    means = transpose(transpose(L)\(L\transpose(m.a)))*Kmn
    covs =  Knn .- transpose(v)*v
    
    return Gaussian(means[:], covs)

end






function get_kldiv(m::SVGPG)
    
    return get_inducing_kldiv(m)
    
end

function get_kldiv(m::SVGPG, x,xmin,xmax,nn)
    
    return get_inducing_kldiv(m)
    
end

function get_inducing_kldiv(m::SVGPG)

    _,mm = size(m.I)
    
    mv = m.a
    Sv = m.L*transpose(m.L)
    
    mp = zeros(1,mm)
    Sp = kernelmatrix(m.kern,m.I)
    
    
    return sum(kldiv(mv,Sv,mp,Sp))

end





function llnormal(x,m,s)
       
    return -0.5 * log(2 * 3.14 * s) - 1/(2*s)*(x-m)^2
    
end

    
function logdetcholesky(X)
    
    m,n = size(X)
    
    return 2 * sum(log.(diag(cholesky(Symmetric(X.+Diagonal(ones(n).*1e-5))).L)))
    
end


function kldiv(m1,S1,m2,S2)
    
    
    _,N = size(m1)
        
    mdiff = m2.-m1
    
    S2L = cholesky(S2.+Diagonal(ones(N).*1e-5))
    S2LL = S2L.L
    S2LU = S2L.U
    
    
    return 0.5*(logdetcholesky(S2)-logdetcholesky(S1) - N + tr(S2LU\(S2LL\S1)) + (mdiff*(S2LU\(S2LL\transpose(mdiff))))[1])

end


function dosample(X,n_sample)
        
    _,N = size(X)
    return randn(n_sample,N)
        
end
Zygote.@nograd dosample

function sample(m::SVGPG, X,n_sample = 10)
    
    model = m(X)
    
    m = Flux.unsqueeze(model.μ,1)
    S = model.Σ
        
    return (dosample(X,n_sample).+m).*Flux.unsqueeze(sqrt.(diag(S)),1)
    
end
    
    
    
    
    
struct ClassifierSVGPG
    
    SVGPG
    
end
Flux.@functor ClassifierSVGPG


applyn(f,X,n_sample) = Flux.unsqueeze(sample(f,X,n_sample),1)

function sample(m::ClassifierSVGPG, X,n_sample = 10)
    
    samples = sample(m.SVGPG,X,n_sample)
    
    return σ.(samples)
    
end



function sample_elbo(m::ClassifierSVGPG,X,y,N,n_sample=10)
    
    sigmoid_samples = sample(m, X,n_sample)
    yaug = y
    
    avg_ll = mean(mean(-log.(sigmoid_samples).*yaug .- log.(1 .- sigmoid_samples).*(1 .- yaug),dims=1))
    kldiv = sum(get_inducing_kldiv(m.SVGPG))
    

    
    return avg_ll + kldiv
    
end